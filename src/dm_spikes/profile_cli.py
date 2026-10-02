"""Scheduler-independent entry point: python -m dm_spikes.profile_cli."""

import argparse
import json
from pathlib import Path
import re
import sys

from .profile_jobs import iter_profiles_radial, run_profile, run_profile_radial, run_profiles


def _write_result(path, result):
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def _profile_paths(configs, directory):
    paths = []
    for index, config in enumerate(configs):
        label = re.sub(r"[^a-zA-Z0-9_-]+", "_", config["id"]).strip("_")[:60] or "profile"
        path = directory / f"{index:03d}_{label}.json"
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON profile or list of profiles")
    parser.add_argument("--id", help="execute only this profile")
    parser.add_argument("--workers", type=int, help="distribute radii among this many spawn workers")
    parser.add_argument("--parallel-axis", choices=("radii",), help=argparse.SUPPRESS)
    parser.add_argument("--block-size", type=int, default=1, help="radii per task for the radial backend")
    parser.add_argument("--inner-threads", type=int, default=1, help="native threads per pool worker")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output", type=Path, help="new output JSON file; existing files are never overwritten")
    outputs.add_argument("--output-dir", type=Path, help="radial backend: save each profile before starting the next")
    args = parser.parse_args(argv)
    radial = args.workers is not None
    if args.parallel_axis is not None and not radial:
        parser.error("--parallel-axis radii requires --workers")
    if not radial and (args.block_size != 1 or args.output_dir is not None):
        parser.error("--block-size and --output-dir require --workers")
    if args.workers is None and args.inner_threads != 1:
        parser.error("--inner-threads configures pool workers; for serial/--id set thread environment variables before Python")
    try:
        with args.manifest.open(encoding="utf-8") as stream:
            configs = json.load(stream)
        if isinstance(configs, dict):
            configs = [configs]
        if not isinstance(configs, list):
            raise ValueError("manifest must contain a profile object or a list")
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"output already exists: {args.output}")
        if args.id is not None:
            selected = [config for config in configs if isinstance(config, dict) and config.get("id") == args.id]
            if len(selected) != 1:
                raise ValueError(f"profile id {args.id!r} must occur exactly once")
            configs = selected
        if args.output_dir is not None:
            paths = _profile_paths(configs, args.output_dir)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            results = iter_profiles_radial(
                configs, max_workers=args.workers, block_size=args.block_size,
                inner_threads=args.inner_threads,
            )
            for index, result in enumerate(results):
                _write_result(paths[index], result)
                print(f"{result['id']} completed in {result['execution']['seconds']:.3f} s; saved: {paths[index]}",
                      flush=True)
            return 0
        if args.id is not None:
            result = (run_profile_radial(
                configs[0], max_workers=args.workers, block_size=args.block_size,
                inner_threads=args.inner_threads,
            ) if radial else run_profile(configs[0]))
        else:
            result = run_profiles(
                configs, backend="radial" if radial else "serial",
                max_workers=args.workers, inner_threads=args.inner_threads, block_size=args.block_size,
            )
        if args.output is None:
            print(json.dumps(result, indent=2, allow_nan=False))
        else:
            # Exclusive create is the race-safe check; exists() is only a fast
            # preflight. Only this coordinating process ever writes the result.
            _write_result(args.output, result)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
