"""Importable worker fixtures; closures are deliberately created only in children."""

import os
from pathlib import Path
import time

from dm_spikes.density_models import hernquist


PARENT_ONLY = False


def synchronized_density(*, directory, participants, expected_threads):
    if PARENT_ONLY:
        raise RuntimeError("spawn unexpectedly inherited mutable parent state")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        if os.environ.get(name) != str(expected_threads):
            raise RuntimeError(f"wrong {name} in worker")
    Path(directory, str(os.getpid())).touch()
    deadline = time.monotonic() + 20
    while len(list(Path(directory).iterdir())) < participants:
        if time.monotonic() > deadline:
            raise RuntimeError("profiles did not execute concurrently")
        time.sleep(0.01)
    return hernquist(total_mass=1.0, scale_radius=1.0)


def failing_density():
    raise ArithmeticError(f"deliberate numerical failure in pid={os.getpid()}")


def exit_worker():
    os._exit(17)


def nesting_density():
    from dm_spikes.profile_jobs import run_profiles
    run_profiles([], backend="process", max_workers=1)
