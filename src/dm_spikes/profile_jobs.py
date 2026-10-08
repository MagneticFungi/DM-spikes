"""Sequential profiles with parallel radial blocks inside each numerical iteration."""

from __future__ import annotations

import importlib
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager

import numpy as np

from .annihilations import rho_spike
from .adiabatic_map import InitialActionTable, build_initial_action_table, solve_initial_energy
from .constants import C_LIGHT, G
from .density_models import HernquistPotential, NFWPotential
from .eddington import TabulatedEddingtonDF, tabulate_eddington_df
from .final_profile import (
    _InnerContinuedPotential, _RadialDensity, _TotalPotential, _initial_state, make_final_profile,
    schwarzschild_radius, solve_self_consistent,
)
from .initial_profile import TabulatedPotential, make_initial_potential, tabulate_potential
from .power_law_profile import cusp_profile
from .pseudo_isothermal_sphere import isothermal_profile


class ProfileConfigurationError(ValueError):
    def __init__(self, profile_id, detail):
        super().__init__(f"Profile {profile_id!r}: {detail}")
        self.profile_id = profile_id


class ProfileExecutionError(RuntimeError):
    def __init__(self, profile_id, detail):
        super().__init__(f"Profile {profile_id!r}: {detail}")
        self.profile_id = profile_id


def _positive(value, name, *, allow_zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")


def _keys(value, allowed, name):
    if type(value) is not dict:
        raise TypeError(f"{name} must be an object")
    extra = value.keys() - allowed
    if extra:
        raise ValueError(f"unknown {name} keys: {sorted(extra)}")


def _prepare(config):
    profile_id = config.get("id", "<unknown>") if isinstance(config, dict) else "<unknown>"
    try:
        # Clone and check that the same finite JSON data reaches every entry point.
        config = json.loads(json.dumps(config, allow_nan=False))
        if type(config) is not dict:
            raise TypeError("profile must be an object")
        engine = config.get("engine", "numerical")
        common = {"id", "engine", "radii", "M_bh", "G", "clight", "annihilation"}
        if engine == "numerical":
            _keys(config, common | {"density", "boundary", "r_ref", "eddington", "solver",
                                    "tabulation"}, "profile")
        elif engine in ("analytic_power_law", "analytic_isothermal"):
            _keys(config, common | {"parameters"}, "profile")
        else:
            raise ValueError(f"unknown engine {engine!r}")
        if type(config.get("id")) is not str or not config["id"].strip():
            raise ValueError("id must be a nonempty string")
        radii = config.get("radii")
        if type(radii) is not list or not radii:
            raise ValueError("radii must be a nonempty list")
        for radius in radii:
            _positive(radius, "radius")
        _positive(config.get("M_bh"), "M_bh", allow_zero=engine == "numerical")
        _positive(config.get("G", G), "G")
        _positive(config.get("clight", C_LIGHT), "clight")
        if "annihilation" in config:
            _keys(config["annihilation"], {"m", "observable_sigma_v", "bh_age"}, "annihilation")
            for name in ("m", "observable_sigma_v"):
                _positive(config["annihilation"].get(name), name)
            _positive(config["annihilation"].get("bh_age", 1e10), "bh_age")
        if engine == "numerical":
            spec = config.get("density")
            _keys(spec, {"factory", "parameters"}, "density")
            factory = spec.get("factory")
            if type(factory) is not str or factory.count(":") != 1:
                raise ValueError("density.factory must be an importable 'module:factory' string")
            module, name = factory.split(":")
            if module in ("__main__", "__mp_main__") or not all(
                item.isidentifier() for item in module.split(".") + name.split(".")
            ):
                raise ValueError("density.factory must identify an importable module")
            if type(spec.get("parameters", {})) is not dict:
                raise TypeError("density.parameters must be an object")
            if config.get("boundary") not in ("finite_escape", "confining"):
                raise ValueError("boundary must be finite_escape or confining")
            if config["boundary"] == "confining":
                _positive(config.get("r_ref"), "r_ref")
            elif config.get("r_ref") is not None:
                raise ValueError("finite_escape does not use r_ref")
            _keys(config.get("eddington", {}),
                  {"escape_slope", "central_potential", "tail_survey_log_radius"}, "eddington")
            if "tabulation" in config:
                _keys(config["tabulation"], {"initial_df_rtol", "initial_df_max_points",
                                              "initial_df_logit_range", "initial_action",
                                              "action_points_per_decade",
                                              "action_circularity_points", "final_potential",
                                              "potential_rtol"}, "tabulation")
                settings = config["tabulation"]
                tolerance = settings.get("initial_df_rtol", 1e-6)
                if type(tolerance) not in (int, float) or not 0.0 < tolerance < 0.01:
                    raise ValueError("tabulation.initial_df_rtol must lie between zero and 0.01")
                count = settings.get("initial_df_max_points", 2049)
                if type(count) is not int or count < 33:
                    raise ValueError("tabulation.initial_df_max_points must be at least 33")
                span = settings.get("initial_df_logit_range", 20.0)
                if type(span) not in (int, float) or not 0.0 < span <= 30.0:
                    raise ValueError("tabulation.initial_df_logit_range must lie between zero and 30")
                if config["boundary"] != "finite_escape":
                    raise ValueError("DF tabulation currently requires finite_escape")
                if (factory not in ("dm_spikes.density_models:nfw",
                                    "dm_spikes.density_models:hernquist")
                        and "central_potential" not in config.get("eddington", {})):
                    raise ValueError("DF tabulation requires a known finite central_potential")
                if type(settings.get("initial_action", False)) is not bool:
                    raise ValueError("tabulation.initial_action must be boolean")
                if type(settings.get("final_potential", False)) is not bool:
                    raise ValueError("tabulation.final_potential must be boolean")
                potential_tolerance = settings.get("potential_rtol", 1e-7)
                if (type(potential_tolerance) not in (int, float)
                        or not 0.0 < potential_tolerance < 0.01):
                    raise ValueError("tabulation.potential_rtol must lie between zero and 0.01")
                if (type(settings.get("action_points_per_decade", 6)) is not int
                        or settings.get("action_points_per_decade", 6) < 2):
                    raise ValueError("tabulation.action_points_per_decade must be at least two")
                if (type(settings.get("action_circularity_points", 25)) is not int
                        or settings.get("action_circularity_points", 25) < 4):
                    raise ValueError("tabulation.action_circularity_points must be at least four")
            solver = config.get("solver", {})
            _keys(solver, {"radial_grid", "rtol", "max_iterations", "epsrel", "limit", "map_kwargs"}, "solver")
            grid = solver.get("radial_grid", radii)
            if type(grid) is not list or len(grid) < 3:
                raise ValueError("solver.radial_grid needs at least three radii")
            for radius in grid:
                _positive(radius, "grid radius")
            if any(b <= a for a, b in zip(grid[:-1], grid[1:])):
                raise ValueError("radial_grid must increase strictly")
            if min(radii) < grid[0] or max(radii) > grid[-1]:
                raise ValueError("output radii must lie within radial_grid")
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
        raise TypeError("density factory must return a callable")
    return density


def _direct_initial_state(config, density):
    """Construct the fixed initial system identically in parent and workers."""
    parameters = config["density"].get("parameters", {})
    potential_factory = {
        "dm_spikes.density_models:nfw": NFWPotential,
        "dm_spikes.density_models:hernquist": HernquistPotential,
    }.get(config["density"]["factory"])
    initial_potential = (potential_factory(**parameters, G=config.get("G", G))
                         if potential_factory is not None else None)
    eddington_options = dict(config.get("eddington", {}))
    if initial_potential is not None:
        if ("central_potential" in eddington_options
                and not math.isclose(float(eddington_options["central_potential"]),
                                     initial_potential.central_potential, rel_tol=1e-10)):
            raise ValueError("central_potential disagrees with the analytic initial potential")
        eddington_options.setdefault("central_potential", initial_potential.central_potential)
    central = eddington_options["central_potential"]
    phi_initial, direct_df = _initial_state(
        density, config["boundary"], config.get("r_ref"), config.get("G", G),
        initial_potential=initial_potential, eddington_kwargs=eddington_options,
    )
    return phi_initial, direct_df, central


def _tabulated_initial_state(config, density, table_data=None, *, sample_evaluator=None):
    """Build a fixed initial DF table once; workers reconstruct its interpolator."""
    phi_initial, direct_df, central = _direct_initial_state(config, density)
    if table_data is None:
        settings = config["tabulation"]
        table = tabulate_eddington_df(
            direct_df, central, rtol=settings.get("initial_df_rtol", 1e-6),
            max_points=settings.get("initial_df_max_points", 2049),
            logit_range=settings.get("initial_df_logit_range", 20.0),
            _sample_evaluator=sample_evaluator,
        )
    else:
        table = TabulatedEddingtonDF(
            central, table_data["logit_nodes"], table_data["log_values"], direct_df,
            table_data["max_validation_error"],
        )
    return phi_initial, table


_table_worker_state = None


def _initialize_table_worker(config):
    global _table_worker_state
    density = _density_from_spec(config["density"])
    potential, direct_df, _ = _direct_initial_state(config, density)
    _table_worker_state = (potential, direct_df)


def _sample_initial_df(energy):
    if _table_worker_state is None:
        raise RuntimeError("initial table worker has not been initialized")
    return float(_table_worker_state[1](energy))


def _sample_initial_energy(job):
    if _table_worker_state is None:
        raise RuntimeError("initial table worker has not been initialized")
    target_action, angular_momentum, options = job
    potential = _table_worker_state[0]
    return solve_initial_energy(
        0.0, angular_momentum, potential, potential,
        _target_action=target_action, **options,
    )


class _InitialTableEvaluator:
    """Bounded parallel batches, restoring input order before interpolation."""

    def __init__(self, executor, workers, report):
        self.executor = executor
        self.workers = workers
        self.report = report

    def _run(self, function, jobs, stage):
        jobs = list(jobs)
        count = len(jobs)
        results = [None] * count
        remaining = iter(enumerate(jobs))
        pending = {}
        completed = 0
        self.report(f"{stage}: 0/{count} muestras; hasta {self.workers} workers")
        last_report = time.perf_counter()

        def submit_one():
            item = next(remaining, None)
            if item is not None:
                index, job = item
                pending[self.executor.submit(function, job)] = index

        try:
            for _ in range(min(2 * self.workers, count)):
                submit_one()
            while pending:
                done, _ = wait(pending, timeout=10.0, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        raise RuntimeError(f"{stage}, muestra {index}: {exc}") from exc
                    completed += 1
                    submit_one()
                now = time.perf_counter()
                if completed == count or now - last_report >= 10.0:
                    self.report(f"{stage}: {completed}/{count} muestras completadas")
                    last_report = now
        except BaseException:
            for future in pending:
                future.cancel()
            raise
        return results

    def df(self, energies):
        return self._run(_sample_initial_df, energies, "Tabla DF inicial")

    def energies(self, targets, options):
        jobs = [(action, momentum, options) for action, momentum in targets]
        return self._run(_sample_initial_energy, jobs, "Tabla de acciones iniciales")


_worker_state = None


def _initialize_radial_worker(config, table_data=None):
    global _worker_state
    # Factories can return local closures; construct them inside each process.
    density = _density_from_spec(config["density"])
    gravity = config.get("G", G)
    if table_data is None:
        phi_initial, initial_df = _initial_state(
            density, config["boundary"], config.get("r_ref"), gravity,
            eddington_kwargs=config.get("eddington", {}),
        )
    else:
        phi_initial, initial_df = _tabulated_initial_state(config, density, table_data)
    action_table = None
    if table_data is not None and "action" in table_data:
        table = table_data["action"]
        action_table = InitialActionTable(
            phi_initial, table["central_potential"], table["log_scales"],
            table["circularities"], table["binding_logits"],
        )
    _worker_state = {
        "config": config, "density": density, "phi_initial": phi_initial,
        "initial_df": initial_df, "iteration": None, "profile": None,
        "action_table": action_table,
    }


def _radial_block(iteration, grid, previous, start, radii, map_options, epsrel, limit,
                  potential_table_data=None):
    state = _worker_state
    if state is None:
        raise RuntimeError("radial worker has not been initialized")
    if state["iteration"] != iteration:
        config = state["config"]
        mass, gravity = config["M_bh"], config.get("G", G)
        if iteration == 1:
            halo = state["phi_initial"]
        else:
            represented = _RadialDensity(grid, previous, state["density"])
            halo = _InnerContinuedPotential(
                make_initial_potential(
                    represented, config["boundary"], G=gravity, r_ref=config.get("r_ref"),
                ),
                state["phi_initial"], float(grid[0]), represented.inner_scale,
            )
            if potential_table_data is not None:
                halo = TabulatedPotential(halo, *potential_table_data)
        potential = (halo if mass == 0.0 else
                     _TotalPotential(halo, mass, gravity,
                                     config.get("clight", C_LIGHT)))
        local_map_options = dict(map_options)
        if state.get("action_table") is not None:
            local_map_options["initial_action_table"] = state["action_table"]
        state["profile"] = make_final_profile(
            state["initial_df"], state["phi_initial"], potential, config["boundary"],
            epsrel=epsrel, limit=limit, map_kwargs=local_map_options,
        )
        state["iteration"] = iteration
    values = []
    for offset, radius in enumerate(radii):
        try:
            values.append(float(state["profile"](radius)))
        except Exception as exc:
            raise RuntimeError(
                f"iteration {iteration}, radius index {start + offset} (r={radius:g}): {exc}"
            ) from exc
    return start, values, os.getpid()


@contextmanager
def _spawn_environment(inner_threads):
    names = (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    )
    old = {name: os.environ.get(name) for name in names}
    old_active = os.environ.get("OMP_MAX_ACTIVE_LEVELS")
    try:
        for name in names:
            os.environ[name] = str(inner_threads)
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if old_active is None:
            os.environ.pop("OMP_MAX_ACTIVE_LEVELS", None)
        else:
            os.environ["OMP_MAX_ACTIVE_LEVELS"] = old_active


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class _RadialEvaluator:
    def __init__(self, executor, workers, block_size):
        self.executor = executor
        self.workers = workers
        self.block_size = block_size
        self.pids = set()

    def __call__(self, iteration, grid, previous, profile, map_options, epsrel, limit):
        # The callback returns only after every radius is filled. The parent
        # then applies Poisson and the convergence test before another iteration.
        values = np.empty(len(grid), dtype=float)
        grid_data, previous_data = grid.tolist(), previous.tolist()
        task_map_options = {key: value for key, value in map_options.items()
                            if key != "initial_action_table"}
        potential = getattr(profile, "potential", None)
        halo = getattr(potential, "halo", potential)
        potential_table_data = (halo.table_data() if isinstance(halo, TabulatedPotential)
                                else None)
        blocks = iter(range(0, len(grid), self.block_size))
        pending = set()

        def submit_one():
            start = next(blocks, None)
            if start is None:
                return False
            stop = min(start + self.block_size, len(grid))
            pending.add(self.executor.submit(
                _radial_block, iteration, grid_data, previous_data, start,
                grid_data[start:stop], task_map_options, epsrel, limit,
                potential_table_data,
            ))
            return True

        for _ in range(min(2 * self.workers, math.ceil(len(grid) / self.block_size))):
            submit_one()
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    start, block, pid = future.result()
                    values[start:start + len(block)] = block
                    self.pids.add(pid)
                    submit_one()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
        return values


def _prepare_initial_tables(config, density, grid, solver, options, *,
                            max_workers, inner_threads, report):
    """Finish the initial tables before starting the radial iteration pool."""
    with ExitStack() as resources:
        evaluator = None
        if max_workers is not None:
            resources.enter_context(_spawn_environment(inner_threads))
            pool = resources.enter_context(ProcessPoolExecutor(
                max_workers=max_workers, mp_context=mp.get_context("spawn"),
                initializer=_initialize_table_worker, initargs=(config,),
            ))
            evaluator = _InitialTableEvaluator(pool, max_workers, report)
        report("Preparando la tabla de la DF inicial")
        phi_initial, initial_df = _tabulated_initial_state(
            config, density, sample_evaluator=evaluator.df if evaluator else None,
        )
        options["initial_potential"] = phi_initial
        options["initial_df"] = initial_df
        table_data = {
            "logit_nodes": initial_df.logit_nodes,
            "log_values": initial_df.log_values,
            "max_validation_error": initial_df.max_validation_error,
        }
        report(f"Tabla DF inicial lista: {len(initial_df.logit_nodes)} nodos")
        settings = config["tabulation"]
        mass, gravity = config["M_bh"], config.get("G", G)
        if settings.get("initial_action", False) and mass > 0.0:
            report("Preparando la tabla inversa de acciones iniciales")
            map_options = dict(solver.get("map_kwargs", {}))
            density_epsrel = solver.get("epsrel", 1e-5)
            map_options.setdefault("action_match_rtol", min(1e-5, density_epsrel / 10.0))
            action_options = dict(map_options.get("radial_action_kwargs", {}))
            action_options.setdefault("epsrel", min(1e-6, density_epsrel / 100.0))
            action_options.setdefault("epsabs", 0.0)
            map_options["radial_action_kwargs"] = action_options
            minimum_scale = 0.01 * math.sqrt(gravity * mass * float(grid[0]))
            maximum_scale = 100.0 * float(grid[-1]) * math.sqrt(
                -2.0 * phi_initial(float(grid[-1]))
            )
            action_table = build_initial_action_table(
                phi_initial, -initial_df.binding_scale, minimum_scale, maximum_scale,
                points_per_decade=settings.get("action_points_per_decade", 6),
                circularity_points=settings.get("action_circularity_points", 25),
                map_kwargs=map_options,
                _energy_evaluator=evaluator.energies if evaluator else None,
            )
            options["map_kwargs"] = dict(solver.get("map_kwargs", {}),
                                          initial_action_table=action_table)
            table_data["action"] = {
                "central_potential": -initial_df.binding_scale,
                "log_scales": action_table.log_scales,
                "circularities": action_table.circularities,
                "binding_logits": action_table.binding_logits,
            }
            report(f"Tabla de acciones iniciales lista: {action_table.binding_logits.size} nodos")
    return table_data


def _execute(config, *, max_workers=None, block_size=1, inner_threads=1,
             on_iteration=None, return_on_max_iterations=False, on_progress=None):
    started = time.perf_counter()
    report = on_progress if on_progress is not None else lambda message: None
    try:
        radii = np.asarray(config["radii"], dtype=float)
        gravity, mass = config.get("G", G), config["M_bh"]
        engine = config.get("engine", "numerical")
        result = {"id": config["id"], "config": config, "radii": radii.tolist()}
        if engine == "numerical":
            solver = dict(config.get("solver", {}))
            grid = solver.pop("radial_grid", config["radii"])
            initial_density = _density_from_spec(config["density"])
            options = dict(
                boundary=config["boundary"], r_ref=config.get("r_ref"), G=gravity,
                clight=config.get("clight", C_LIGHT),
                eddington_kwargs=config.get("eddington", {}),
                iteration_callback=on_iteration,
                return_on_max_iterations=return_on_max_iterations, **solver,
            )
            table_data = None
            if "tabulation" in config:
                table_data = _prepare_initial_tables(
                    config, initial_density, grid, solver, options,
                    max_workers=max_workers, inner_threads=inner_threads, report=report,
                )
                settings = config["tabulation"]
                if settings.get("final_potential", False):
                    potential_rtol = settings.get("potential_rtol", 1e-7)

                    def tabulator(halo, radial_grid):
                        report("Actualizando el potencial de Poisson y su tabla radial")
                        table = tabulate_potential(
                            halo, float(radial_grid[0]),
                            float(radial_grid[-1]) * 100.0,
                            rtol=potential_rtol,
                        )
                        report("Tabla del potencial actualizada")
                        return table

                    options["_potential_tabulator"] = tabulator
            if max_workers is None:
                report("Iniciando las iteraciones: cálculo serial de radios")
                equilibrium = solve_self_consistent(initial_density, grid, mass, **options)
            else:
                workers = min(max_workers, math.ceil(len(grid) / block_size))
                report(f"Iniciando las iteraciones: {workers} workers para bloques de radios")
                with _spawn_environment(inner_threads):
                    with ProcessPoolExecutor(
                        max_workers=workers, mp_context=mp.get_context("spawn"),
                        initializer=_initialize_radial_worker, initargs=(config, table_data),
                    ) as pool:
                        evaluator = _RadialEvaluator(pool, workers, block_size)
                        equilibrium = solve_self_consistent(
                            initial_density, grid, mass, _density_evaluator=evaluator,
                            **options,
                        )
                result["parallel"] = {
                    "workers_requested": max_workers, "workers_allocated": workers,
                    "worker_pids": sorted(evaluator.pids), "block_size": block_size,
                    "inner_threads": inner_threads,
                }
            values = np.array([equilibrium.density_profile(float(r)) for r in radii])
            result["self_consistent"] = {
                "converged": equilibrium.converged,
                "history": equilibrium.history,
                "radial_grid": equilibrium.radii.tolist(),
                "density_grid": equilibrium.density.tolist(),
                "density_potential_grid": [
                    equilibrium.density_potential(float(r)) for r in equilibrium.radii
                ],
                "potential_grid": [equilibrium.potential(float(r)) for r in equilibrium.radii],
            }
        else:
            radius_s = schwarzschild_radius(mass, G=gravity, clight=config.get("clight", C_LIGHT))
            parameters = config.get("parameters", {})
            if engine == "analytic_power_law":
                values = np.asarray(cusp_profile(radii, mass, R_S=radius_s, **parameters))
            else:
                values = np.asarray(isothermal_profile(
                    radii, mass, R_S=radius_s, G=gravity, **parameters,
                ))
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ArithmeticError("profile returned an invalid density")
        result["rho_prime"] = values.tolist()
        if "annihilation" in config:
            result["rho_spike"] = np.asarray(rho_spike(values, **config["annihilation"])).tolist()
        result["execution"] = {"pid": os.getpid(), "seconds": time.perf_counter() - started}
        return result
    except Exception as exc:
        raise ProfileExecutionError(config["id"], f"{type(exc).__name__}: {exc}") from exc


def run_profile(config, *, max_workers=None, block_size=1, inner_threads=1,
                on_iteration=None, return_on_max_iterations=False, on_progress=None):
    """Parallelize initial table samples and radial blocks with max_workers.

    Optional on_progress receives stage and initial-table progress messages
    in the parent process. Profiles and self-consistency iterations remain
    sequential; Poisson updates and interpolation refinement are coordinated
    by the parent.
    """
    if max_workers is not None:
        _positive_integer(max_workers, "max_workers")
    _positive_integer(block_size, "block_size")
    _positive_integer(inner_threads, "inner_threads")
    if on_iteration is not None and not callable(on_iteration):
        raise TypeError("on_iteration must be callable")
    if on_progress is not None and not callable(on_progress):
        raise TypeError("on_progress must be callable")
    if type(return_on_max_iterations) is not bool:
        raise TypeError("return_on_max_iterations must be boolean")
    return _execute(_prepare(config), max_workers=max_workers,
                    block_size=block_size, inner_threads=inner_threads,
                    on_iteration=on_iteration,
                    return_on_max_iterations=return_on_max_iterations,
                    on_progress=on_progress)


def run_profiles(configs, *, backend=None, max_workers=None,
                 block_size=1, inner_threads=1):
    """Run profiles in order; radial workers finish before the next profile."""
    if backend is None:
        backend = "radial" if max_workers is not None else "serial"
    if backend not in ("serial", "radial"):
        raise ValueError("backend must be serial or radial")
    if backend == "radial" and max_workers is None:
        raise ValueError("radial backend requires max_workers")
    if backend == "serial" and max_workers is not None:
        raise ValueError("serial backend cannot specify max_workers")
    if max_workers is not None:
        _positive_integer(max_workers, "max_workers")
    _positive_integer(block_size, "block_size")
    _positive_integer(inner_threads, "inner_threads")
    jobs = [_prepare(config) for config in configs]
    ids = [job["id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("profile ids in a batch must be unique")
    return [_execute(job, max_workers=max_workers, block_size=block_size,
                     inner_threads=inner_threads) for job in jobs]


def run_profile_radial(config, *, max_workers, block_size=1, inner_threads=1):
    """Compatibility entry point for one numerical profile with radial workers."""
    prepared = _prepare(config)
    if prepared.get("engine", "numerical") != "numerical":
        raise ProfileConfigurationError(prepared["id"], "radial backend requires engine='numerical'")
    return run_profile(prepared, max_workers=max_workers, block_size=block_size,
                       inner_threads=inner_threads)


def iter_profiles_radial(configs, *, max_workers, block_size=1, inner_threads=1):
    """Yield complete numerical profiles in order, closing each pool first."""
    _positive_integer(max_workers, "max_workers")
    _positive_integer(block_size, "block_size")
    _positive_integer(inner_threads, "inner_threads")
    jobs = [_prepare(config) for config in configs]
    ids = [job["id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("profile ids in a batch must be unique")
    for job in jobs:
        if job.get("engine", "numerical") != "numerical":
            raise ProfileConfigurationError(job["id"], "radial backend requires engine='numerical'")
    for job in jobs:
        yield _execute(job, max_workers=max_workers, block_size=block_size,
                       inner_threads=inner_threads)


__all__ = [
    "run_profile", "run_profiles", "run_profile_radial", "iter_profiles_radial",
    "ProfileConfigurationError", "ProfileExecutionError",
]
