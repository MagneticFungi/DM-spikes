"""Run numerical equilibrium or analytical profiles from a JSON manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

from .profile_jobs import run_profile, run_profiles


def _positive_int(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _write_result(path, result):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON profile or list of profiles")
    parser.add_argument("--id", help="execute only this profile")
    parser.add_argument("--workers", type=_positive_int,
                        help="CPU processes for radial blocks of one numerical profile")
    parser.add_argument("--block-size", type=_positive_int, default=1,
                        help="radii per worker task (default: 1)")
    parser.add_argument("--inner-threads", type=_positive_int, default=1,
                        help="native math threads per worker (default: 1)")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output", type=Path, help="write a new JSON output file")
    outputs.add_argument("--output-dir", type=Path, help="save each profile as a new JSON file")
    args = parser.parse_args(argv)
    try:
        with args.manifest.open(encoding="utf-8") as stream:
            configs = json.load(stream)
        if isinstance(configs, dict):
            configs = [configs]
        if not isinstance(configs, list) or not configs:
            raise ValueError("manifest must contain an object or a nonempty list of objects")
        if args.id is not None:
            configs = [item for item in configs if isinstance(item, dict) and item.get("id") == args.id]
            if len(configs) != 1:
                raise ValueError(f"profile id {args.id!r} must occur exactly once")
        if args.output_dir is None:
            options = dict(max_workers=args.workers, block_size=args.block_size,
                           inner_threads=args.inner_threads)
            result = (run_profile(configs[0], **options) if len(configs) == 1 else
                      run_profiles(configs, backend="radial" if args.workers else "serial",
                                   **options))
            if args.output is None:
                print(json.dumps(result, indent=2, allow_nan=False))
            else:
                _write_result(args.output, result)
            return 0
        paths = []
        for index, config in enumerate(configs):
            label = re.sub(r"[^a-zA-Z0-9_-]+", "_", config["id"]).strip("_")[:60] or "profile"
            path = args.output_dir / f"{index:03d}_{label}.json"
            if path.exists():
                raise FileExistsError(path)
            paths.append(path)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for config, path in zip(configs, paths):
            result = run_profile(config, max_workers=args.workers,
                                 block_size=args.block_size,
                                 inner_threads=args.inner_threads)
            _write_result(path, result)
            print(f"{result['id']} saved: {path}", flush=True)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
