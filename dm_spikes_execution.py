"""Compute four final dark-matter profiles with the numerical DF pipeline.

Each profile uses 300 radii from 4.001 Schwarzschild radii to its domain end.
The Gondolo-Silk spike radius only sets that end. Model-specific initial
potentials and derivatives provide stable inputs; densities come from the
numerical Eddington inversion, adiabatic map and final phase-space integral.
Profiles run in order, with their radii distributed among spawned processes. A completed
profile is saved to its own NPZ before the next profile starts.

Run after installing the package with ``python -m pip install -e .``:

    python dm_spikes_execution.py --workers 2
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

# Set native-thread defaults before importing NumPy/SciPy in parent or children.
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import numpy as np

from dm_spikes.constants import G
from dm_spikes.density_models import nfw, power_law
from dm_spikes.eddington import make_eddington_df
from dm_spikes.final_profile import make_final_profile, schwarzschild_radius
from dm_spikes.power_law_profile import D, rho_D, spike_radius


RADII_COUNT = 300
DEFAULT_M_BH = 4.0e6  # Msun; change with --m-bh if needed.
DEFAULT_NFW_R_S = 20.0e3  # pc; illustrative NFW scale radius.
_WORKER_SPEC = None
_WORKER_DENSITY = None


class _PowerLawPotential:
    """Initial cusp potential with phi(0)=0, including its exact derivatives."""

    def __init__(self, rho0: float, r0: float, gamma: float):
        self.exponent = 2.0 - gamma
        self.slope_scale = 4.0 * math.pi * G * rho0 * r0**gamma / (3.0 - gamma)
        self.potential_scale = self.slope_scale / self.exponent

    def __call__(self, radius: float) -> float:
        return self.potential_scale * radius**self.exponent

    def prime(self, radius: float) -> float:
        return self.slope_scale * radius ** (self.exponent - 1.0)

    def second(self, radius: float) -> float:
        return self.prime(radius) * (self.exponent - 1.0) / radius


class _NFWPotential:
    """Initial NFW potential with phi(infinity)=0 and stable inner derivatives."""

    def __init__(self, rho_s: float, r_s: float):
        self.r_s = r_s
        self.density_scale = 4.0 * math.pi * G * rho_s
        self.potential_scale = self.density_scale * r_s**2

    @staticmethod
    def _mass_factor(x: float) -> float:
        # [log(1+x)-x/(1+x)]/x² loses digits for small x.
        if x < 0.05:
            return math.fsum(
                (-1.0)**k * (k + 1.0) / (k + 2.0) * x**k
                for k in range(18)
            )
        return (math.log1p(x) - x / (1.0 + x)) / x**2

    @staticmethod
    def _mass_factor_prime(x: float) -> float:
        if x < 0.05:
            return math.fsum(
                (-1.0)**k * k * (k + 1.0) / (k + 2.0) * x ** (k - 1)
                for k in range(1, 18)
            )
        numerator = math.log1p(x) - x / (1.0 + x)
        return 1.0 / (x * (1.0 + x) ** 2) - 2.0 * numerator / x**3

    def __call__(self, radius: float) -> float:
        x = radius / self.r_s
        return -self.potential_scale * math.log1p(x) / x

    def prime(self, radius: float) -> float:
        return self.density_scale * self.r_s * self._mass_factor(radius / self.r_s)

    def second(self, radius: float) -> float:
        return self.density_scale * self._mass_factor_prime(radius / self.r_s)


def _positive_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _specifications(m_bh: float, nfw_r_s: float, epsrel: float):
    r_schwarzschild = schwarzschild_radius(m_bh)
    lower_radius = 4.001 * r_schwarzschild
    profiles = []
    reference_spike_radius = None

    for gamma, name in (
        (1.0, "power_law_gamma_1"),
        (0.5, "power_law_gamma_0p5"),
        (1.5, "power_law_gamma_1p5"),
    ):
        rho_at_d = float(rho_D(gamma))
        r_spike = float(spike_radius(m_bh, gamma, rho_at_d, D))
        if not math.isfinite(r_spike) or r_spike <= lower_radius:
            raise ValueError(f"invalid spike radius for gamma={gamma:g}")
        if gamma == 1.0:
            reference_spike_radius = r_spike
        profiles.append({
            "name": name,
            "model": "power_law",
            "gamma": gamma,
            "rho0": rho_at_d,
            "r0": float(D),
            "rho_at_D": rho_at_d,
            "M_bh": m_bh,
            "R_S": r_schwarzschild,
            "r_max": r_spike,
            "epsrel": epsrel,
            "radii": np.geomspace(lower_radius, r_spike, RADII_COUNT),
        })

    # Match the NFW density at D to the gamma=1 power law. The scale radius
    # remains an independent model choice, here 20 kpc by default.
    x_d = D / nfw_r_s
    rho_reference = float(rho_D(1.0))
    rho_s = rho_reference * x_d * (1.0 + x_d) ** 2
    profiles.append({
        "name": "nfw",
        "model": "nfw",
        "rho_s": rho_s,
        "r_s": nfw_r_s,
        "rho_at_D": rho_reference,
        "M_bh": m_bh,
        "R_S": r_schwarzschild,
        "r_max": reference_spike_radius,
        "epsrel": epsrel,
        "radii": np.geomspace(lower_radius, reference_spike_radius, RADII_COUNT),
    })
    return profiles


def _build_final_density(spec):
    """Build the numerical Eddington DF, adiabatic map and final integral."""
    if spec["model"] == "power_law":
        density = power_law(rho0=spec["rho0"], r0=spec["r0"], gamma=spec["gamma"])
        boundary = "confining"
        potential = _PowerLawPotential(spec["rho0"], spec["r0"], spec["gamma"])
        central_potential = 0.0
    else:
        density = nfw(rho_s=spec["rho_s"], r_s=spec["r_s"])
        boundary = "finite_escape"
        potential = _NFWPotential(spec["rho_s"], spec["r_s"])
        central_potential = -potential.potential_scale

    initial_df = make_eddington_df(
        density, potential, boundary,
        density_prime=density.prime,
        density_second=density.second,
        potential_prime=potential.prime,
        potential_second=potential.second,
        central_potential=central_potential,
    )
    return make_final_profile(
        initial_df, potential, spec["M_bh"], epsrel=spec["epsrel"]
    )


def _init_worker(spec):
    global _WORKER_SPEC, _WORKER_DENSITY
    _WORKER_SPEC = spec
    _WORKER_DENSITY = None


def _calculate_block(start, radii):
    global _WORKER_DENSITY
    location = f"radii[{start}:{start + len(radii)}]"
    try:
        if _WORKER_DENSITY is None:
            _WORKER_DENSITY = _build_final_density(_WORKER_SPEC)
        values = []
        for offset, radius in enumerate(radii):
            location = f"radius index {start + offset}, r={radius:.17g} pc"
            value = float(_WORKER_DENSITY(float(radius)))
            if not math.isfinite(value) or value < 0.0:
                raise ArithmeticError("final density must be finite and nonnegative")
            values.append(value)
        return start, values
    except Exception as exc:
        raise RuntimeError(f"{_WORKER_SPEC['name']}, {location}: {exc}") from exc


def _compute_profile(spec, workers: int, block_size: int) -> np.ndarray:
    radii = spec["radii"]
    worker_count = min(workers, math.ceil(len(radii) / block_size))
    if sys.platform == "win32" and worker_count > 61:
        raise ValueError("Windows ProcessPoolExecutor supports at most 61 workers")
    worker_spec = {key: value for key, value in spec.items() if key != "radii"}
    values = np.full(len(radii), np.nan)
    starts = iter(range(0, len(radii), block_size))

    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(worker_spec,),
    ) as pool:
        pending = {}

        def submit_next():
            start = next(starts, None)
            if start is not None:
                block = radii[start:start + block_size]
                pending[pool.submit(_calculate_block, start, block)] = start

        try:
            for _ in range(2 * worker_count):
                submit_next()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    start = pending.pop(future)
                    result_start, block_values = future.result()
                    if result_start != start:
                        raise RuntimeError("worker returned a block at the wrong index")
                    values[start:start + len(block_values)] = block_values
                for _ in done:
                    submit_next()
        finally:
            for future in pending:
                future.cancel()

    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ArithmeticError(f"{spec['name']}: incomplete or invalid density grid")
    return values


def _save_profile(path: Path, spec, densities: np.ndarray, elapsed: float):
    payload = {
        "radii": spec["radii"],
        "rho_prime": densities,
        "radius_unit": "pc",
        "density_unit": "Msun/pc^3",
        "model": spec["model"],
        "M_bh": spec["M_bh"],
        "R_S": spec["R_S"],
        "D": float(D),
        "rho_at_D": spec["rho_at_D"],
        "r_max": spec["r_max"],
        "epsrel": spec["epsrel"],
        "elapsed_seconds": elapsed,
    }
    for key in ("gamma", "rho0", "r0", "rho_s", "r_s"):
        if key in spec:
            payload[key] = spec[key]
    # Publish the NPZ only after the compressed archive is complete.
    temporary_path = path.with_name(path.name + ".part")
    try:
        with temporary_path.open("xb") as stream:
            np.savez_compressed(stream, **payload)
        if path.exists():
            raise FileExistsError(path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=_positive_int, default=2,
                        help="radial worker processes per profile (default: 2)")
    parser.add_argument("--block-size", type=_positive_int, default=1,
                        help="radii per worker task (default: 1)")
    parser.add_argument("--m-bh", type=_positive_float, default=DEFAULT_M_BH,
                        help="black-hole mass in Msun (default: 4e6)")
    parser.add_argument("--nfw-rs", type=_positive_float, default=DEFAULT_NFW_R_S,
                        help="NFW scale radius in pc (default: 20000)")
    parser.add_argument("--epsrel", type=_positive_float, default=1e-3,
                        help="relative tolerance of the final density integral (default: 1e-3)")
    parser.add_argument("--output-dir", type=Path,
                        help="new directory for four NPZ files (default: timestamped under results)")
    args = parser.parse_args(argv)
    if args.epsrel <= 5e-14:
        parser.error("--epsrel must exceed the float64 quadrature floor (5e-14)")

    profiles = _specifications(args.m_bh, args.nfw_rs, args.epsrel)
    output_dir = args.output_dir or Path("results") / (
        "dm_spikes_numeric_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"output directory already exists: {output_dir}")

    print(f"Resultados: {output_dir.resolve()}", flush=True)
    for spec in profiles:
        label = spec["name"]
        output_file = output_dir / f"{label}.npz"
        print(f"Calculando {label} ({RADII_COUNT} radios)...", flush=True)
        started = time.perf_counter()
        try:
            densities = _compute_profile(spec, args.workers, args.block_size)
            elapsed = time.perf_counter() - started
            _save_profile(output_file, spec, densities, elapsed)
        except Exception:
            print(f"Falló {label}; los perfiles anteriores siguen guardados.",
                  file=sys.stderr, flush=True)
            raise
        print(f"{label} listo en {elapsed:.2f} s. Guardado en {output_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
