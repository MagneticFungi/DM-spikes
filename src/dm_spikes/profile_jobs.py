"""Serial and spawn execution across JSON-configured profiles or their radii.

No scientific callable or cache crosses a process boundary. Use an importable
``module:factory`` for a custom initial density. See docs/parallel_profiles.md.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
import importlib
import json
import math
import multiprocessing as mp
import os
import threading
import time

import numpy as np

from .annihilations import rho_spike
from .constants import C_LIGHT, G
from .eddington import make_eddington_df
from .final_profile import make_final_profile, schwarzschild_radius
from .initial_profile import make_initial_potential
from .power_law_profile import cusp_profile
from .pseudo_isothermal_sphere import isothermal_profile


class ProfileConfigurationError(ValueError):
    """Invalid JSON or job configuration, identified before pool creation."""

    def __init__(self, profile_id, detail):
        self.profile_id = profile_id
        self.detail = detail
        super().__init__(profile_id, detail)

    def __str__(self):
        return f"Profile {self.profile_id!r}: {self.detail}"


class ProfileExecutionError(RuntimeError):
    """A profile failed; the original exception is retained as the cause."""

    def __init__(self, profile_id, detail):
        self.profile_id = profile_id
        self.detail = detail
        super().__init__(profile_id, detail)

    def __str__(self):
        return f"Profile {self.profile_id!r}: {self.detail}"


_THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
_ENV_LOCK = threading.Lock()
_IN_PROFILE = False


def _check_json(value):
    # JSON encoders otherwise silently coerce integer keys and tuples. Reject
    # those too, so the serial and process routes see exactly the same data.
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _check_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _check_json(item)
        return
    raise TypeError("configuration must contain only finite JSON data (no callables or arrays)")


def _positive(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _keys(options, allowed, name):
    if type(options) is not dict:
        raise TypeError(f"{name} must be an object")
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {name} keys: {sorted(unknown)}")


def _prepare(config):
    profile_id = config.get("id", "<unknown>") if isinstance(config, dict) else "<unknown>"
    try:
        _check_json(config)
        config = json.loads(json.dumps(config, allow_nan=False))
        common = {"id", "radii", "M_bh", "G", "clight", "engine", "annihilation"}
        engine = config.get("engine", "numerical")
        if engine == "numerical":
            extra = {"density", "boundary", "r_ref", "eddington", "final",
                     "density_derivatives", "potential_derivatives"}
        elif engine in ("analytic_power_law", "analytic_isothermal"):
            extra = {"parameters"}
        else:
            raise ValueError(f"unknown engine {engine!r}")
        _keys(config, common | extra, "profile")
        if type(config.get("id")) is not str or not config["id"].strip():
            raise ValueError("id must be a nonempty string")
        radii = config["radii"]
        if type(radii) is not list or not radii:
            raise ValueError("radii must be a nonempty one-dimensional JSON list")
        for radius in radii:
            _positive(radius, "radius")
        for name, default in (("M_bh", None), ("G", G), ("clight", C_LIGHT)):
            _positive(config.get(name, default), name)
        if "annihilation" in config:
            _keys(config["annihilation"], {"m", "observable_sigma_v", "bh_age"}, "annihilation")
            for name in ("m", "observable_sigma_v"):
                _positive(config["annihilation"][name], name)
            _positive(config["annihilation"].get("bh_age", 1e10), "bh_age")
        if engine == "numerical":
            spec = config["density"]
            _keys(spec, {"factory", "parameters"}, "density")
            factory = spec["factory"]
            if not isinstance(factory, str) or factory.count(":") != 1:
                raise ValueError("density.factory must be an importable 'module:factory' string")
            module, name = factory.split(":")
            if module in ("__main__", "__mp_main__") or not all(
                part.isidentifier() for part in module.split(".") + name.split(".")
            ):
                raise ValueError("density.factory must live in an importable module, not __main__")
            if type(spec.get("parameters", {})) is not dict:
                raise ValueError("density.parameters must be an object")
            boundary = config["boundary"]
            if boundary not in ("finite_escape", "confining"):
                raise ValueError("boundary must be finite_escape or confining")
            if boundary == "confining":
                _positive(config.get("r_ref"), "r_ref")
            elif config.get("r_ref") is not None:
                raise ValueError("finite_escape does not use r_ref")
            _keys(config.get("eddington", {}),
                  {"escape_slope", "central_potential", "tail_survey_log_radius"}, "eddington")
            _keys(config.get("final", {}), {"epsabs", "epsrel", "limit", "map_kwargs"}, "final")
            for flag in ("density_derivatives", "potential_derivatives"):
                if type(config.get(flag, False)) is not bool:
                    raise ValueError(f"{flag} must be boolean")
        else:
            allowed = ({"gamma", "rho0", "r0", "rtol", "max_terms"}
                       if engine == "analytic_power_law" else {"rho0", "sigma_v"})
            _keys(config.get("parameters", {}), allowed, "parameters")
        return config
    except Exception as exc:
        raise ProfileConfigurationError(profile_id, str(exc)) from exc


def _density_from_spec(spec):
    module, name = spec["factory"].split(":")
    factory = importlib.import_module(module)
    for part in name.split("."):
        factory = getattr(factory, part)
    density = factory(**spec.get("parameters", {}))
    if not callable(density):
        raise TypeError("density factory must return a scalar callable")
    return density


def _build_numerical_profile(config):
    """Construct the unchanged scalar numerical pipeline in the calling process."""
    density = _density_from_spec(config["density"])
    potential = make_initial_potential(
        density, config["boundary"], G=config.get("G", G), r_ref=config.get("r_ref")
    )
    eddington_options = dict(config.get("eddington", {}))
    if config.get("density_derivatives", False):
        eddington_options.update(density_prime=density.prime, density_second=density.second)
    if config.get("potential_derivatives", False):
        eddington_options.update(potential_prime=potential.prime, potential_second=potential.second)
    distribution = make_eddington_df(density, potential, config["boundary"], **eddington_options)
    return make_final_profile(
        distribution, potential, config["M_bh"], G=config.get("G", G),
        clight=config.get("clight", C_LIGHT), **config.get("final", {})
    )


def _density_result(config, values):
    radii = np.asarray(config["radii"], dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape != radii.shape or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ArithmeticError("profile returned invalid density values")
    result = {"id": config["id"], "config": config, "radii": radii.tolist(),
              "rho_prime": values.tolist()}
    if "annihilation" in config:
        result["rho_spike"] = np.asarray(rho_spike(values, **config["annihilation"])).tolist()
    return result


def _execute(config):
    global _IN_PROFILE
    previous = _IN_PROFILE
    _IN_PROFILE = True
    started = time.perf_counter()
    try:
        radii = np.asarray(config["radii"], dtype=float)
        mass = config["M_bh"]
        gravity, light = config.get("G", G), config.get("clight", C_LIGHT)
        engine = config.get("engine", "numerical")
        if engine == "numerical":
            profile = _build_numerical_profile(config)
            values = np.asarray([profile(float(r)) for r in radii])
        else:
            radius_s = schwarzschild_radius(mass, G=gravity, clight=light)
            parameters = config.get("parameters", {})
            if engine == "analytic_power_law":
                values = np.asarray(cusp_profile(radii, mass, R_S=radius_s, **parameters))
            else:
                values = np.asarray(isothermal_profile(radii, mass, R_S=radius_s, G=gravity, **parameters))
        result = _density_result(config, values)
        result["execution"] = {"pid": os.getpid(), "seconds": time.perf_counter() - started}
        return result
    except Exception as exc:
        raise ProfileExecutionError(config["id"], f"{type(exc).__name__}: {exc}") from exc
    finally:
        _IN_PROFILE = previous


def run_profile(config):
    """Run one JSON-configured profile locally; suitable for a scheduler task."""
    return _execute(_prepare(config))


@contextmanager
def _spawn_environment(inner_threads):
    # Must be set BEFORE spawn: an initializer runs after imports of numpy/scipy.
    # Serialize this runner's environment changes and restore even on failures.
    with _ENV_LOCK:
        limits = {key: str(inner_threads) for key in _THREAD_ENV}
        limits["OMP_MAX_ACTIVE_LEVELS"] = "1"
        saved = {key: os.environ.get(key) for key in limits}
        try:
            os.environ.update(limits)
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def run_profiles(configs, *, backend="serial", max_workers=None, inner_threads=1, block_size=1):
    """Return results in input order, with a fresh callable/cache per profile.

    ``backend='serial'`` evaluates every profile in the calling process.
    ``backend='radial'`` processes numerical profiles one at a time and
    distributes their radii among spawned workers. See ``iter_profiles_radial``
    to consume/save each completed profile. Call the radial route under a
    __main__ guard in an importable script. ``inner_threads`` limits native
    libraries in the workers, not those already loaded in the serial caller.
    """
    if backend == "radial":
        return list(iter_profiles_radial(
            configs, max_workers=max_workers, inner_threads=inner_threads, block_size=block_size
        ))
    if backend != "serial":
        raise ValueError("backend must be 'serial' or 'radial'")
    if type(block_size) is not int or block_size != 1:
        raise ValueError("block_size applies only to the radial backend")
    if max_workers is not None:
        raise ValueError("max_workers applies only to the radial backend")
    if type(inner_threads) is not int or inner_threads != 1:
        raise ValueError("inner_threads configures radial workers; for serial execution set the environment before Python")
    jobs = [_prepare(config) for config in configs]
    ids = set()
    for job in jobs:
        if job["id"] in ids:
            raise ProfileConfigurationError(job["id"], "duplicate profile id in batch")
        ids.add(job["id"])
    return [_execute(job) for job in jobs]


# These objects exist only in a spawned radial worker. Initializing lazily in
# its first task lets construction errors travel through a Future with their
# original traceback, instead of killing the pool initializer.
_RADIAL_CONFIG = None
_RADIAL_PROFILE = None


def _init_radial_worker(config):
    global _RADIAL_CONFIG, _RADIAL_PROFILE, _IN_PROFILE
    _RADIAL_CONFIG = config
    _RADIAL_PROFILE = None
    _IN_PROFILE = True


def _radial_block(start, radii):
    global _RADIAL_PROFILE
    location = f"building profile for radii[{start}:{start + len(radii)}]"
    try:
        if _RADIAL_PROFILE is None:
            _RADIAL_PROFILE = _build_numerical_profile(_RADIAL_CONFIG)
        values = []
        for offset, radius in enumerate(radii):
            location = f"radius index {start + offset}, r={radius!r}"
            value = float(_RADIAL_PROFILE(radius))
            if not math.isfinite(value) or value < 0:
                raise ArithmeticError("profile returned invalid density values")
            values.append(value)
        return values, os.getpid()
    except Exception as exc:
        raise ProfileExecutionError(
            _RADIAL_CONFIG["id"], f"{location}: {type(exc).__name__}: {exc}"
        ) from exc


def _check_radial_resources(max_workers, block_size, inner_threads):
    if _IN_PROFILE or mp.current_process().name != "MainProcess":
        raise RuntimeError("nested process pools are not supported")
    for name, value in (("max_workers", max_workers), ("block_size", block_size),
                        ("inner_threads", inner_threads)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be an explicit positive integer")


def _run_radial_prepared(config, max_workers, block_size, inner_threads):
    started = time.perf_counter()
    radii = config["radii"]
    worker_count = min(max_workers, (len(radii) + block_size - 1) // block_size)
    # The full grid and output remain in the parent. A worker receives the
    # profile description once and only a small block of radii per subsequent task.
    worker_config = {key: value for key, value in config.items() if key != "radii"}
    values = [None] * len(radii)
    worker_pids = set()
    try:
        with _spawn_environment(inner_threads):
            with ProcessPoolExecutor(
                max_workers=worker_count, mp_context=mp.get_context("spawn"),
                initializer=_init_radial_worker, initargs=(worker_config,),
            ) as pool:
                starts = iter(range(0, len(radii), block_size))
                pending = {}

                def submit_next():
                    start = next(starts, None)
                    if start is not None:
                        pending[pool.submit(_radial_block, start, radii[start:start + block_size])] = start

                try:
                    # Bound submission/memory independently of the grid length.
                    for _ in range(2 * worker_count):
                        submit_next()
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            start = pending.pop(future)
                            block, pid = future.result()
                            values[start:start + len(block)] = block
                            worker_pids.add(pid)
                        for _ in done:
                            submit_next()
                finally:
                    for future in pending:
                        future.cancel()
        result = _density_result(config, values)
        result["execution"] = {
            "backend": "radial", "pid": os.getpid(), "worker_pids": sorted(worker_pids),
            "requested_workers": max_workers, "max_workers": worker_count,
            "block_size": block_size, "inner_threads": inner_threads,
            "seconds": time.perf_counter() - started,
        }
        return result
    except ProfileExecutionError:
        raise
    except BrokenProcessPool as exc:
        raise ProfileExecutionError(
            config["id"], "radial process pool terminated; the profile is incomplete"
        ) from exc
    except Exception as exc:
        raise ProfileExecutionError(config["id"], f"{type(exc).__name__}: {exc}") from exc


def run_profile_radial(config, *, max_workers, block_size=1, inner_threads=1):
    """Evaluate one numerical profile using a spawn pool over its radii.

    Each worker builds the original scalar pipeline once and retains its caches
    between tasks. ``block_size=1`` schedules individual radii dynamically; larger
    values group adjacent input entries without changing their quadratures.
    A single worker still uses a child process. Returned radii retain input order.
    """
    return next(iter_profiles_radial(
        [config], max_workers=max_workers, block_size=block_size, inner_threads=inner_threads
    ))


def iter_profiles_radial(configs, *, max_workers, block_size=1, inner_threads=1):
    """Yield fully computed numerical profiles, one at a time, in input order.

    All JSON/configuration validation precedes the first pool. The current pool
    is closed before yielding its result; the next profile does not start until
    the caller asks for the next result. This permits immediate per-profile saves
    and guarantees there is at most one pool, with fresh state per profile.
    Analytical engines should use their existing vectorized serial route.
    """
    _check_radial_resources(max_workers, block_size, inner_threads)
    jobs = [_prepare(config) for config in configs]
    ids = set()
    for job in jobs:
        if job.get("engine", "numerical") != "numerical":
            raise ProfileConfigurationError(job["id"], "radial backend requires engine='numerical'")
        if job["id"] in ids:
            raise ProfileConfigurationError(job["id"], "duplicate profile id in batch")
        ids.add(job["id"])
    for job in jobs:
        yield _run_radial_prepared(job, max_workers, block_size, inner_threads)


__all__ = ["run_profile", "run_profiles", "run_profile_radial", "iter_profiles_radial",
           "ProfileConfigurationError", "ProfileExecutionError"]
