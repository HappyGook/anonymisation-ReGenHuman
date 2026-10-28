#!/usr/bin/env python3

"""
Example usage:
python scripts/sample_videos.py \
    --input-dir assets/examples \
    --count 5 \
    --seed 42

"""

import argparse
import random
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Randomly select distinct videos from a directory."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be greater than zero")

    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    videos = sorted(
        path.resolve()
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if len(videos) < args.count:
        print(
            f"Requested {args.count} videos, but found only {len(videos)}.",
            file=sys.stderr,
        )
        return 1

    selected = random.Random(args.seed).sample(videos, args.count)

    for video in selected:
        print(video)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())