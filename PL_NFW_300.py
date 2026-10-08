"""Compute three power-law and one NFW self-consistent profile in sequence."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from dm_spikes.final_profile import schwarzschild_radius
from dm_spikes.power_law_profile import D, rho_D, spike_radius
from dm_spikes.profile_jobs import run_profile


def _positive_float(raw):
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value


def _positive_int(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _specifications(mass, nfw_scale, count, rtol, max_iterations):
    lower = 4.001 * schwarzschild_radius(mass)
    configs = []
    gamma_one_radius = None
    for gamma, label in ((1.0, "power_law_gamma_1"),
                         (0.5, "power_law_gamma_0p5"),
                         (1.5, "power_law_gamma_1p5")):
        rho0 = float(rho_D(gamma))
        outer = float(spike_radius(mass, gamma, rho0, D))
        if gamma == 1.0:
            gamma_one_radius = outer
        configs.append({
            "id": label, "engine": "numerical", "M_bh": mass,
            "radii": np.geomspace(lower, outer, count).tolist(),
            "density": {"factory": "dm_spikes.density_models:power_law",
                        "parameters": {"rho0": rho0, "r0": float(D), "gamma": gamma}},
            "boundary": "confining", "r_ref": float(D),
            "solver": {"rtol": rtol, "max_iterations": max_iterations},
        })
    x = D / nfw_scale
    rho_s = float(rho_D(1.0)) * x * (1.0 + x)**2
    configs.append({
        "id": "nfw", "engine": "numerical", "M_bh": mass,
        "radii": np.geomspace(lower, gamma_one_radius, count).tolist(),
        "density": {"factory": "dm_spikes.density_models:nfw",
                    "parameters": {"rho_s": rho_s, "r_s": nfw_scale}},
        "boundary": "finite_escape",
        "solver": {"rtol": rtol, "max_iterations": max_iterations},
    })
    return configs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-bh", type=_positive_float, default=4e6)
    parser.add_argument("--nfw-rs", type=_positive_float, default=2e4)
    parser.add_argument("--radii-count", type=_positive_int, default=300)
    parser.add_argument("--rtol", type=_positive_float, default=1e-4)
    parser.add_argument("--max-iterations", type=_positive_int, default=100)
    parser.add_argument("--workers", type=_positive_int,
                        help="CPU processes for each profile's radial grid")
    parser.add_argument("--block-size", type=_positive_int, default=1)
    parser.add_argument("--inner-threads", type=_positive_int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.radii_count < 3:
        parser.error("at least three radii are required")
    destination = args.output_dir or Path("results") / (
        "self_consistent_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    destination.mkdir(parents=True, exist_ok=False)
    for config in _specifications(args.m_bh, args.nfw_rs, args.radii_count,
                                  args.rtol, args.max_iterations):
        result = run_profile(config, max_workers=args.workers,
                             block_size=args.block_size,
                             inner_threads=args.inner_threads)
        path = destination / f"{config['id']}.npz"
        np.savez_compressed(
            path, radii=result["radii"], rho_prime=result["rho_prime"],
            potential=result["self_consistent"]["potential_grid"],
            config_json=json.dumps(config),
            history_json=json.dumps(result["self_consistent"]["history"]),
            radius_unit="pc", density_unit="Msun/pc^3",
        )
        print(f"{config['id']}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
