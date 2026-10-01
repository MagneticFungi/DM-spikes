"""Real spawn processes, serial equivalence, and scheduler CLI contracts."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

import profile_job_fixtures
from dm_spikes.profile_jobs import (
    ProfileConfigurationError, ProfileExecutionError, run_profile, run_profiles,
)


ROOT = Path(__file__).resolve().parents[1]


def analytic_jobs():
    jobs = json.loads((ROOT / "examples/profile_jobs.json").read_text())
    for gamma in (0.5, 1.8):
        job = deepcopy(jobs[0])
        job["id"] = f"power-{gamma}"
        job["parameters"]["gamma"] = gamma
        job["radii"] = [0.1, 0.000001, 0.001, 0.1]  # retain unsorted/repeated radii
        jobs.append(job)
    return jobs


def numerical_job(factory, **parameters):
    return {
        "id": "custom", "M_bh": 1.0, "G": 1.0, "clight": 10.0,
        "radii": [0.04, 0.08], "boundary": "finite_escape",
        "density": {"factory": factory, "parameters": parameters},
    }


def assert_equivalent(expected, actual):
    assert [item["id"] for item in actual] == [item["id"] for item in expected]
    for left, right in zip(expected, actual):
        assert left["config"] == right["config"]
        assert left["radii"] == right["radii"]
        np.testing.assert_allclose(left["rho_prime"], right["rho_prime"], rtol=2e-12, atol=0)
        if "rho_spike" in left:
            np.testing.assert_allclose(left["rho_spike"], right["rho_spike"], rtol=2e-12, atol=0)


@pytest.mark.parametrize("workers", [1, 2])
def test_spawn_analytic_matches_serial_and_scalar_apis(workers):
    from dm_spikes.annihilations import rho_spike
    from dm_spikes.final_profile import schwarzschild_radius
    from dm_spikes.power_law_profile import cusp_profile
    from dm_spikes.pseudo_isothermal_sphere import isothermal_profile

    jobs = analytic_jobs()
    untouched = deepcopy(jobs)
    serial = run_profiles(jobs)
    parallel = run_profiles(jobs, backend="process", max_workers=workers)
    assert_equivalent(serial, parallel)
    assert jobs == untouched
    assert all(item["execution"]["pid"] == os.getpid() for item in serial)
    assert all(item["execution"]["pid"] != os.getpid() for item in parallel)
    for config, result in zip(jobs, parallel):
        function = cusp_profile if config["engine"] == "analytic_power_law" else isothermal_profile
        direct = function(config["radii"], config["M_bh"],
                          R_S=schwarzschild_radius(config["M_bh"]), **config["parameters"])
        np.testing.assert_allclose(result["rho_prime"], direct, rtol=2e-12, atol=0)
        if "annihilation" in config:
            np.testing.assert_allclose(result["rho_spike"], rho_spike(direct, **config["annihilation"]))


@pytest.mark.parametrize("workers,threads", [(1, 2), (2, 1)])
def test_spawn_builds_closures_concurrently_limits_threads_and_does_not_inherit_state(
    tmp_path, monkeypatch, workers, threads,
):
    monkeypatch.setattr(profile_job_fixtures, "PARENT_ONLY", True)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    monkeypatch.delenv("BLIS_NUM_THREADS", raising=False)
    before = dict(os.environ)
    jobs = []
    for index in range(workers):
        config = numerical_job("profile_job_fixtures:synchronized_density",
                               directory=str(tmp_path), participants=workers, expected_threads=threads)
        config["id"] = str(index)
        jobs.append(config)
    results = run_profiles(jobs, backend="process", max_workers=workers, inner_threads=threads)
    assert len({item["execution"]["pid"] for item in results}) == workers
    assert dict(os.environ) == before
    assert [item["id"] for item in results] == [str(i) for i in range(workers)]
    assert all(item["rho_prime"] == [0.0, 0.0] for item in results)


@pytest.mark.parametrize("factory,match", [
    ("failing_density", "deliberate numerical failure"),
    ("exit_worker", "pool terminated.*unresolved"),
    ("nesting_density", "nested process pools"),
])
def test_worker_errors_keep_profile_identity_and_restore_environment(factory, match):
    before = dict(os.environ)
    config = numerical_job(f"profile_job_fixtures:{factory}")
    with pytest.raises(ProfileExecutionError, match=match) as caught:
        run_profiles([config], backend="process", max_workers=2)
    assert caught.value.profile_id == "custom"
    assert caught.value.__cause__ is not None
    assert dict(os.environ) == before
    if factory == "failing_density":
        assert f"pid={os.getpid()}" not in str(caught.value)
        with pytest.raises(ProfileExecutionError, match=match):
            run_profile(config)
    # A failed pool must not poison a subsequent call.
    assert run_profiles(analytic_jobs()[:1], backend="process", max_workers=1)[0]["id"]


@pytest.mark.parametrize("bad", [lambda r: r, np.array([1.0]), float("nan"), {1: "value"}])
def test_non_json_configuration_is_rejected_before_any_worker(tmp_path, bad):
    first = numerical_job("profile_job_fixtures:synchronized_density",
                          directory=str(tmp_path), participants=1, expected_threads=1)
    invalid = deepcopy(first)
    invalid["id"] = "invalid-json"
    invalid["density"]["parameters"]["bad"] = bad
    with pytest.raises(ProfileConfigurationError, match="invalid-json.*JSON"):
        run_profiles([first, invalid], backend="process", max_workers=2)
    assert list(tmp_path.iterdir()) == []


def test_duplicate_identity_rejected_before_process_work():
    config = analytic_jobs()[0]
    with pytest.raises(ProfileConfigurationError, match="duplicate"):
        run_profiles([config, config], backend="process", max_workers=2)


def test_builtin_numerical_factories_construct_in_spawn_and_preserve_capture():
    jobs = json.loads((ROOT / "examples/numerical_profiles.json").read_text())
    for job in jobs:
        job["radii"] = [0.08, 0.04, 0.08]
    serial = run_profiles(jobs)
    spawned = run_profiles(jobs, backend="process", max_workers=2)
    assert_equivalent(serial, spawned)
    assert all(result["execution"]["pid"] != os.getpid() for result in spawned)
    assert all(result["rho_prime"] == [0, 0, 0] for result in spawned)


@pytest.mark.parametrize("kwargs", [{}, {"max_workers": 0}, {"max_workers": True},
                                   {"max_workers": 1, "inner_threads": 0}])
def test_process_resource_configuration_is_explicit(kwargs):
    with pytest.raises(ValueError):
        run_profiles(analytic_jobs(), backend="process", **kwargs)


def cli(*args):
    return subprocess.run([sys.executable, "-B", "-m", "dm_spikes.profile_cli", *map(str, args)],
                          capture_output=True, text=True, timeout=60)


def test_scheduler_single_id_pool_cli_and_exclusive_output(tmp_path):
    manifest = ROOT / "examples/profile_jobs.json"
    target = tmp_path / "result.json"
    completed = cli(manifest, "--id", "power-law-analytic", "--output", target)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(target.read_text())
    assert result["id"] == "power-law-analytic"
    assert_equivalent(run_profiles(analytic_jobs()[:1]), [result])
    original = target.read_bytes()
    rejected = cli(manifest, "--id", "isothermal-analytic", "--output", target)
    assert rejected.returncode != 0 and "already exists" in rejected.stderr
    assert target.read_bytes() == original
    missing = cli(manifest, "--id", "missing")
    assert missing.returncode != 0 and "missing" in missing.stderr
    wrong_threads = cli(manifest, "--id", "power-law-analytic", "--inner-threads", 2)
    assert wrong_threads.returncode != 0 and "before Python" in wrong_threads.stderr
    pooled = cli(manifest, "--workers", 2)
    assert pooled.returncode == 0, pooled.stderr
    assert_equivalent(run_profiles(analytic_jobs()[:2]), json.loads(pooled.stdout))


def test_worker_failure_cli_has_nonzero_status_and_no_result(tmp_path):
    config = numerical_job("dm_spikes.density_models:hernquist", total_mass=-1, scale_radius=1)
    path = tmp_path / "bad.json"
    target = tmp_path / "bad-result.json"
    path.write_text(json.dumps(config))
    failed = cli(path, "--workers", 1, "--output", target)
    assert failed.returncode != 0 and "custom" in failed.stderr
    assert not target.exists()


def test_independent_scheduler_processes_cannot_write_the_same_output(tmp_path):
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    config = numerical_job("profile_job_fixtures:synchronized_density",
                           directory=str(barrier), participants=2, expected_threads=1)
    manifest = tmp_path / "job.json"
    manifest.write_text(json.dumps(config))
    target = tmp_path / "shared.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[key] = "1"
    command = [sys.executable, "-B", "-m", "dm_spikes.profile_cli", str(manifest),
               "--id", "custom", "--output", str(target)]
    children = [subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True) for _ in range(2)]
    try:
        outputs = [child.communicate(timeout=40) for child in children]
        assert sorted(child.returncode for child in children) == [0, 1], outputs
        result = json.loads(target.read_text())
        assert result["id"] == "custom"
        assert result["rho_prime"] == [0.0, 0.0]
        assert len(list(barrier.iterdir())) == 2  # both passed the existence preflight
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()


@pytest.mark.skipif(os.environ.get("DM_SPIKES_RUN_SLOW") != "1",
                    reason="set DM_SPIKES_RUN_SLOW=1 for the direct numerical quadrature (minutes)")
def test_direct_numerical_profiles_outside_capture_match_spawn(tmp_path):
    # Full unmodified Eddington -> action map -> double phase-space integral.
    # A deliberately loose 5% quadrature tolerance keeps this integration test
    # bounded; the serial/process agreement is checked much more tightly.
    template = json.loads((ROOT / "examples/numerical_profiles.json").read_text())[2]
    template["final"] = {"epsrel": 0.05}
    jobs = []
    for gamma in (0.75, 0.6):
        config = deepcopy(template)
        config["id"] = f"numerical-gamma-{gamma}"
        config["density"]["parameters"]["gamma"] = gamma
        jobs.append(config)
    started = time.perf_counter()
    serial = run_profiles(jobs)
    serial_seconds = time.perf_counter() - started
    print(f"direct numerical serial: {serial_seconds:.3f}s", flush=True)
    started = time.perf_counter()
    parallel = run_profiles(jobs, backend="process", max_workers=2)
    parallel_seconds = time.perf_counter() - started
    print(f"direct numerical spawn(2): {parallel_seconds:.3f}s", flush=True)
    assert_equivalent(serial, parallel)
    assert len({item["execution"]["pid"] for item in parallel}) == 2
    for result in parallel:
        assert result["rho_prime"][0] == 0  # true capture boundary
        assert result["rho_prime"][1] > 0   # actually exercises every numerical stage
    (tmp_path / "numerical_equivalence.json").write_text(json.dumps({
        "serial_seconds": serial_seconds, "parallel_seconds": parallel_seconds,
        "serial": serial, "parallel": parallel,
    }, indent=2))
