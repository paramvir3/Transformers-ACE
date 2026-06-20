#!/usr/bin/env python3
"""Stream every Nth frame from a trajectory into a new extxyz file."""

import argparse
from pathlib import Path

from ase.io import iread, write


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input trajectory")
    parser.add_argument("output", type=Path, help="Output extxyz trajectory")
    parser.add_argument("--every", type=int, default=10, help="Keep every Nth frame")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.every < 1:
        raise ValueError("--every must be at least 1")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input trajectory not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)

    frames_read = 0
    frames_written = 0
    for index, atoms in enumerate(iread(args.input, index=":")):
        frames_read += 1
        if index % args.every == 0:
            write(args.output, atoms, format="extxyz", append=True)
            frames_written += 1

    print(f"Read {frames_read} frames")
    print(f"Wrote {frames_written} frames to {args.output.resolve()}")


if __name__ == "__main__":
    main()
