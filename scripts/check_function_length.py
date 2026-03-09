"""Check Python function length thresholds and enforce a hard limit.

The count is based on physical source lines from `lineno` to `end_lineno`,
but the leading function docstring block is excluded entirely. That means
doctests inside the function docstring do not count toward the limit.

Examples:
    >>> is_too_long(401)
    True
    >>> is_too_long(400)
    False
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LIMIT = 400


@dataclass(frozen=True)
class FunctionLength:
    """Measured function length metadata."""

    path: Path
    qualname: str
    lines: int


def is_too_long(lines: int, limit: int = LIMIT) -> bool:
    """Return True if a function exceeds the hard limit.

    Examples:
        >>> is_too_long(400)
        False
        >>> is_too_long(401)
        True
    """
    return lines > limit


def iter_python_files(root: Path) -> Iterable[Path]:
    """Yield repository Python files excluding common build artifacts.

    Examples:
        >>> any(path.name == "main.py" for path in iter_python_files(Path(".")))
        True
    """
    excluded = {".git", ".venv", "build", "dist", "__pycache__"}
    for path in root.rglob("*.py"):
        if any(part in excluded for part in path.parts):
            continue
        yield path


def _docstring_span(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[int, int] | None:
    """Return the leading docstring span for a function if present.

    Examples:
        >>> tree = ast.parse("def f():\\n    \\'\\'\\'doc\\'\\'\\'\\n    return 1\\n")
        >>> _docstring_span(tree.body[0])
        (2, 2)
    """
    if not node.body:
        return None
    first_stmt = node.body[0]
    if not isinstance(first_stmt, ast.Expr):
        return None
    value = first_stmt.value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return None
    return getattr(first_stmt, "lineno", 0), getattr(first_stmt, "end_lineno", 0)


def counted_function_lines(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    """Return counted function lines excluding the leading docstring span.

    Examples:
        >>> tree = ast.parse(
        ...     "def f():\\n"
        ...     "    \\'\\'\\'doc\\n"
        ...     "    >>> 1\\n"
        ...     "    \\'\\'\\'\\n"
        ...     "    x = 1\\n"
        ...     "    return x\\n"
        ... )
        >>> counted_function_lines(tree.body[0])
        3
    """
    start = getattr(node, "lineno", 0)
    end = getattr(node, "end_lineno", start)
    total = max(0, end - start + 1)
    span = _docstring_span(node)
    if span is None:
        return total
    doc_start, doc_end = span
    return max(0, total - max(0, doc_end - doc_start + 1))


class _FunctionCollector(ast.NodeVisitor):
    """Collect qualified function lengths from an AST."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.scope: list[str] = []
        self.functions: list[FunctionLength] = []

    def _record(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        qualname = ".".join([*self.scope, node.name]) if self.scope else node.name
        self.functions.append(
            FunctionLength(
                path=self.path,
                qualname=qualname,
                lines=counted_function_lines(node),
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node)
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def collect_lengths_from_source(source: str, path: Path) -> list[FunctionLength]:
    """Collect function lengths from Python source text.

    Examples:
        >>> funcs = collect_lengths_from_source("def f():\\n    return 1\\n", Path("x.py"))
        >>> funcs[0].qualname, funcs[0].lines
        ('f', 2)
    """
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    collector = _FunctionCollector(path)
    collector.visit(tree)
    return collector.functions


def collect_lengths(path: Path) -> list[FunctionLength]:
    """Collect function lengths from a file.

    Examples:
        >>> any(item.qualname == "main" for item in collect_lengths(Path("main.py")))
        True
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return collect_lengths_from_source(source, path)


def format_error(func_len: FunctionLength, limit: int = LIMIT) -> str:
    """Format an error line.

    Examples:
        >>> format_error(FunctionLength(Path("x.py"), "A.f", 401))
        'ERROR: x.py:A.f has 401 counted lines (limit 400)'
    """
    return (
        f"ERROR: {func_len.path}:{func_len.qualname} has {func_len.lines} "
        f"counted lines (limit {limit})"
    )


def main() -> int:
    """Run the function length check.

    Examples:
        >>> callable(main)
        True
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=LIMIT,
        help="Maximum counted lines per function.",
    )
    args = parser.parse_args()

    errors = 0
    for path in iter_python_files(args.root):
        for func_len in collect_lengths(path):
            if is_too_long(func_len.lines, args.limit):
                print(format_error(func_len, args.limit))
                errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
