"""CLI wrapper for the SegEdge pipeline."""

from __future__ import annotations


def main() -> None:
    """Run the SegEdge pipeline entrypoint.

    Examples:
        >>> isinstance(main.__name__, str)
        True
    """
    from segedge.pipeline.run import main as _main

    _main()


if __name__ == "__main__":
    main()
