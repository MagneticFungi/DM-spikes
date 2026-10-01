"""Scheduler-independent entry point: python -m dm_spikes.profile_cli."""

import argparse
import json
from pathlib import Path
import sys

from .profile_jobs import run_profile, run_profiles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON profile or list of profiles")
    parser.add_argument("--id", help="execute only this profile, in this process")
    parser.add_argument("--workers", type=int, help="use a local spawn pool of this size")
    parser.add_argument("--inner-threads", type=int, default=1, help="native threads per pool worker")
    parser.add_argument("--output", type=Path, help="new output JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.id is not None and args.workers is not None:
        parser.error("--id executes a single profile; omit --workers")
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
            result = run_profile(selected[0])
        else:
            result = run_profiles(
                configs, backend="serial" if args.workers is None else "process",
                max_workers=args.workers, inner_threads=args.inner_threads,
            )
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            # Exclusive create is the race-safe check; exists() is only a fast
            # preflight. Only this coordinating process ever writes the result.
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
