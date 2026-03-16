"""CLI wrapper for the SegEdge pipeline."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    """Run the SegEdge pipeline entrypoint.

    Examples:
        >>> callable(main)
        True
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", help="Explicit config YAML path.")
    args, remaining = parser.parse_known_args()
    if args.config:
        os.environ["SEGEDGE_CONFIG"] = args.config
    sys.argv = [sys.argv[0], *remaining]
    from segedge.pipeline.run import main as _main

    _main()


if __name__ == "__main__":
    main()
