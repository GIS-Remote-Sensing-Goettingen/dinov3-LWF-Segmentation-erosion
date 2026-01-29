"""CLI wrapper for the split evaluation pipeline."""

from __future__ import annotations


def main() -> None:
    """Run the split evaluation entrypoint.

    Examples:
        >>> callable(main)
        True
    """
    from segedge.pipeline.split_eval import main as _main

    _main()


if __name__ == "__main__":
    main()
