"""Reproducible end-to-end serial/spawn benchmark, including pool startup.

Run from the repository root after installing dm-spikes. A numerical manifest
can take many minutes: every final-density sample performs the original map
and Eddington quadrature. No interpolation or monkeypatching is used.
"""

import argparse
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

# Set before numerical imports, for a comparable serial baseline as well.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

import numpy as np
import scipy

from dm_spikes.profile_jobs import run_profiles


def parent_peak_rss_bytes():
    """Process lifetime peak only, NOT aggregate parent+workers or incremental RSS."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize
        return None
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 1:
        parser.error("repeats and workers must be positive")
    if args.manifest:
        jobs = json.loads(args.manifest.read_text(encoding="utf-8"))
        if isinstance(jobs, dict):
            jobs = [jobs]
    else:
        templates = json.loads((Path(__file__).resolve().parents[1] / "examples/profile_jobs.json").read_text())
        jobs = []
        for index in range(8):
            job = deepcopy(templates[index % 2])
            job["id"] = f"analytic-{index}"
            job["radii"] = np.geomspace(1e-6, 1.0, 256).tolist()
            if index % 2 == 0:
                job["parameters"]["gamma"] = 0.6 + 0.15 * index
            jobs.append(job)
    report = {
        "platform": platform.platform(), "processor": platform.processor(),
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "visible_cpus": os.cpu_count(), "start_method": "spawn", "inner_threads": 1,
        "profile_count": len(jobs), "radii_counts": [len(job["radii"]) for job in jobs],
        "configs": jobs, "runs": [],
        "memory_note": "parent lifetime peak RSS only; workers and aggregate pool memory excluded",
    }
    reference = None
    for backend, workers in (("serial", None), ("process", 1), ("process", args.workers)):
        durations = []
        differences = []
        samples = []
        for repeat in range(args.repeats):
            started = time.perf_counter()
            results = run_profiles(jobs, backend=backend, max_workers=workers)
            elapsed = time.perf_counter() - started
            if reference is None:
                reference = results
            difference = 0.0
            for expected, actual in zip(reference, results):
                assert expected["config"] == actual["config"]
                for key in ("rho_prime", "rho_spike"):
                    if key not in expected:
                        continue
                    left, right = np.asarray(expected[key]), np.asarray(actual[key])
                    np.testing.assert_allclose(left, right, rtol=2e-6, atol=0)
                    difference = max(difference, float(np.max(np.abs(left - right) /
                                     np.maximum(np.abs(left), np.finfo(float).tiny))))
            durations.append(elapsed)
            differences.append(difference)
            samples.append([{"id": result["id"], **result["execution"]} for result in results])
            print(f"{backend} workers={workers} repeat={repeat + 1}: {elapsed:.6f}s", file=sys.stderr, flush=True)
        report["runs"].append({
            "backend": backend, "workers": workers, "seconds": durations,
            "median_seconds": statistics.median(durations), "max_relative_difference": max(differences),
            "parent_peak_rss_bytes": parent_peak_rss_bytes(), "samples": samples,
        })
    report["reference_densities"] = [
        {key: value for key, value in result.items() if key in ("id", "rho_prime", "rho_spike")}
        for result in reference
    ]
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
