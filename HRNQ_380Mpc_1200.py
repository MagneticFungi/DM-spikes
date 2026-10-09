"""Compute the self-consistent Hernquist profile from 4.001 R_S to 380 Mpc.

NPZ names include the local date and time at the start of the run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

import numpy as np

from dm_spikes.constants import C_LIGHT, G
from dm_spikes.final_profile import schwarzschild_radius
from dm_spikes.profile_jobs import run_profile


M_HALO = 6.06e11  # Msun
SCALE_RADIUS = 40_000.0  # pc = 40 kpc
M_BH = 2.6e6  # Msun
OUTER_RADIUS = 380.0e6  # pc = 380 Mpc
DENSITY_EPSREL = 1e-5


def _positive_int(raw):
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _iterations_limit(raw):
    value = _positive_int(raw)
    if value > 100:
        raise argparse.ArgumentTypeError("max-iterations cannot exceed 100")
    return value


def _tolerance(raw):
    value = float(raw)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("rtol must be between zero and one")
    return value


def _specification(count, rtol, max_iterations):
    return {
        "id": "hernquist_380mpc", "engine": "numerical", "M_bh": M_BH,
        "G": G, "clight": C_LIGHT,
        "radii": np.geomspace(
            4.001 * schwarzschild_radius(M_BH, G=G, clight=C_LIGHT),
            OUTER_RADIUS, count,
        ).tolist(),
        "density": {
            "factory": "dm_spikes.density_models:hernquist",
            "parameters": {"total_mass": M_HALO, "scale_radius": SCALE_RADIUS},
        },
        "boundary": "finite_escape",
        # Hernquist: Phi_i(r) = -G M_halo / (r + a), so Phi_i(0) = -G M_halo / a.
        # The library uses the analytic initial potential and numerical Poisson
        # for the updated halo in subsequent iterations.
        "eddington": {"central_potential": -G * M_HALO / SCALE_RADIUS},
        "tabulation": {
            "initial_df_rtol": 1e-6, "initial_action": True,
            "final_potential": True, "potential_rtol": 1e-7,
        },
        "solver": {
            "rtol": rtol, "epsrel": DENSITY_EPSREL,
            "max_iterations": max_iterations,
        },
    }


def _save_npz(path, **arrays):
    """Publish a complete NPZ after writing it to a temporary file."""
    temporary = path.with_name(path.name + ".part")
    if path.exists():
        raise FileExistsError(path)
    with temporary.open("xb") as stream:
        try:
            np.savez_compressed(stream, **arrays)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        if path.exists():
            raise FileExistsError(path)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _save_iteration(destination, state, config, elapsed_seconds, run_stamp, started_at):
    record = state.history[-1]
    iteration = record["iteration"]
    path = destination / f"hernquist_380mpc_{run_stamp}_iter_{iteration:03d}.npz"
    return _save_npz(
        path,
        radii=state.radii, rho_prime=state.density,
        iteration=iteration,
        max_relative_density_change=record["max_relative_density_change"],
        converged=state.converged,
        elapsed_seconds=elapsed_seconds, started_at=started_at,
        history_json=json.dumps(state.history), config_json=json.dumps(config),
        radius_unit="pc", density_unit="Msun/pc^3",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radii-count", type=_positive_int, default=1200)
    parser.add_argument("--rtol", type=_tolerance, default=1e-4)
    parser.add_argument("--max-iterations", type=_iterations_limit, default=100)
    parser.add_argument("--workers", type=_positive_int,
                        help="CPU processes for radial blocks; omitted means serial")
    parser.add_argument("--block-size", type=_positive_int, default=1)
    parser.add_argument("--inner-threads", type=_positive_int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.radii_count < 3:
        parser.error("at least three radii are required")
    if DENSITY_EPSREL > args.rtol / 8.0:
        parser.error("rtol must be at least 8e-5 with density epsrel=1e-5")

    config = _specification(args.radii_count, args.rtol, args.max_iterations)
    run_started = datetime.now().astimezone()
    run_stamp = run_started.strftime("%Y%m%d_%H%M%S")
    started_at = run_started.isoformat()
    destination = args.output_dir or Path("results") / (
        "self_consistent_hernquist_380mpc_" + run_stamp
    )
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / f"hernquist_380mpc_{run_stamp}_final.npz"
    mode = f"{args.workers} workers" if args.workers is not None else "modo serial"
    print(f"Calculando Hernquist ({args.radii_count} radios, {mode}, "
          f"máximo {args.max_iterations} iteraciones). Salida: {destination}", flush=True)
    started = time.perf_counter()

    def on_iteration(state):
        saved = _save_iteration(
            destination, state, config, time.perf_counter() - started,
            run_stamp, started_at,
        )
        record = state.history[-1]
        print(f"Iteración {record['iteration']:03d}: cambio máximo "
              f"{record['max_relative_density_change']:.6g}; guardado en {saved}",
              flush=True)

    result = run_profile(
        config, max_workers=args.workers, block_size=args.block_size,
        inner_threads=args.inner_threads, on_iteration=on_iteration,
        return_on_max_iterations=True,
    )
    state = result["self_consistent"]
    elapsed = time.perf_counter() - started
    _save_npz(
        path,
        radii=result["radii"], rho_prime=result["rho_prime"],
        potential=state["potential_grid"],
        density_potential=state["density_potential_grid"],
        converged=state["converged"], iterations=len(state["history"]),
        elapsed_seconds=elapsed, started_at=started_at,
        config_json=json.dumps(config), history_json=json.dumps(state["history"]),
        radius_unit="pc", density_unit="Msun/pc^3", potential_unit="(km/s)^2",
    )
    if state["converged"]:
        status = f"Hernquist convergió en {len(state['history'])} iteraciones"
    else:
        status = f"Hernquist alcanzó el límite de {args.max_iterations} iteraciones sin converger"
    print(f"{status} en {timedelta(seconds=round(elapsed))} "
          f"({elapsed:.1f} s). Guardado en {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
