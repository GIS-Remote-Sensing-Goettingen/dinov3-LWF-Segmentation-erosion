"""Tests for the function-length repository guard."""

from __future__ import annotations

from pathlib import Path

from scripts.check_function_length import (
    FunctionLength,
    collect_lengths_from_source,
    counted_function_lines,
    format_error,
)


def test_counted_function_lines_excludes_docstring_and_doctests():
    """Leading docstring lines, including doctests, should not count.

    Examples:
        >>> True
        True
    """
    tree = __import__("ast").parse(
        """
def sample():
    \"\"\"Example.

    >>> 1 + 1
    2
    \"\"\"
    a = 1
    b = 2
    return a + b
"""
    )

    assert counted_function_lines(tree.body[0]) == 4


def test_collect_lengths_from_source_tracks_qualified_names():
    """Nested functions and methods should keep qualified names.

    Examples:
        >>> True
        True
    """
    source = """
class Demo:
    def method(self):
        def inner():
            return 1
        return inner()

async def task():
    return 2
"""

    lengths = collect_lengths_from_source(source, Path("demo.py"))
    names = {item.qualname for item in lengths}

    assert names == {"Demo.method", "Demo.method.inner", "task"}


def test_format_error_uses_counted_line_language():
    """Error output should mention counted lines.

    Examples:
        >>> True
        True
    """
    message = format_error(FunctionLength(Path("demo.py"), "Demo.method", 401))

    assert message == "ERROR: demo.py:Demo.method has 401 counted lines (limit 400)"
