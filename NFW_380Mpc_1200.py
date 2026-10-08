"""Compute the self-consistent NFW profile from 4.001 R_S to 380 Mpc."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

import numpy as np

from dm_spikes.constants import C_LIGHT
from dm_spikes.final_profile import schwarzschild_radius
from dm_spikes.profile_jobs import run_profile


M_BH = 2.6e6  # Msun
NFW_R_S = 20_000.0  # pc
NFW_RHO_S = 0.003568  # Msun/pc^3
OUTER_RADIUS = 380.0e6  # pc = 380 Mpc


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
        "id": "nfw_380mpc", "engine": "numerical", "M_bh": M_BH,
        "clight": C_LIGHT,
        "radii": np.geomspace(
            4.001 * schwarzschild_radius(M_BH, clight=C_LIGHT), OUTER_RADIUS, count,
        ).tolist(),
        "density": {
            "factory": "dm_spikes.density_models:nfw",
            "parameters": {"rho_s": NFW_RHO_S, "r_s": NFW_R_S},
        },
        "boundary": "finite_escape",
        "tabulation": {
            "initial_df_rtol": 1e-6, "initial_action": True,
            "final_potential": True, "potential_rtol": 1e-7,
        },
        "solver": {"rtol": rtol, "max_iterations": max_iterations},
    }


def _save_iteration(destination, state, config, elapsed_seconds):
    """Publish one complete density profile without leaving a partial NPZ."""
    record = state.history[-1]
    iteration = record["iteration"]
    path = destination / f"nfw_380mpc_iter_{iteration:03d}.npz"
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                radii=state.radii,
                rho_prime=state.density,
                iteration=iteration,
                max_relative_density_change=record["max_relative_density_change"],
                converged=state.converged,
                elapsed_seconds=elapsed_seconds,
                history_json=json.dumps(state.history),
                config_json=json.dumps(config),
                radius_unit="pc",
                density_unit="Msun/pc^3",
            )
        if path.exists():
            raise FileExistsError(path)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


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

    config = _specification(args.radii_count, args.rtol, args.max_iterations)
    destination = args.output_dir or Path("results") / (
        "self_consistent_nfw_380mpc_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "nfw_380mpc.npz"
    mode = f"{args.workers} workers" if args.workers is not None else "modo serial"
    print(f"Calculando NFW ({args.radii_count} radios, {mode}, "
          f"máximo {args.max_iterations} iteraciones). Salida: {destination}", flush=True)
    started = time.perf_counter()

    def on_iteration(state):
        saved = _save_iteration(destination, state, config, time.perf_counter() - started)
        record = state.history[-1]
        print(f"Iteración {record['iteration']:03d}: cambio máximo "
              f"{record['max_relative_density_change']:.6g}; guardado en {saved}",
              flush=True)

    result = run_profile(
        config, max_workers=args.workers, block_size=args.block_size,
        inner_threads=args.inner_threads, on_iteration=on_iteration,
        return_on_max_iterations=True,
        on_progress=lambda message: print(message, flush=True),
    )
    state = result["self_consistent"]
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        path,
        radii=result["radii"], rho_prime=result["rho_prime"],
        potential=state["potential_grid"],
        converged=state["converged"],
        iterations=len(state["history"]),
        elapsed_seconds=elapsed,
        config_json=json.dumps(config),
        history_json=json.dumps(state["history"]),
        radius_unit="pc", density_unit="Msun/pc^3",
    )
    if state["converged"]:
        status = f"NFW convergió en {len(state['history'])} iteraciones"
    else:
        status = f"NFW alcanzó el límite de {args.max_iterations} iteraciones sin converger"
    print(f"{status} en {timedelta(seconds=round(elapsed))} "
          f"({elapsed:.1f} s). Guardado en {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
