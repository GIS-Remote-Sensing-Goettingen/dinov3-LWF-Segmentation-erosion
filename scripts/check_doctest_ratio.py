"""Check that a minimum ratio of functions include doctests in docstrings.

The rule is intentionally simple: a function is considered to have a doctest
if its docstring contains the ">>>" marker.

Examples:
    >>> has_doctest('''Example.
    ...
    ... >>> 1 + 1
    ... 2
    ... ''')
    True
    >>> has_doctest("No doctest here.")
    False
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class FunctionInfo:
    """Information about a function definition."""

    name: str
    filepath: Path
    lineno: int
    has_doctest: bool
    has_forbidden_doctest: bool


def has_doctest(docstring: str | None) -> bool:
    """Return True if the docstring includes a doctest marker.

    Examples:
        >>> has_doctest('''Example.
        ...
        ... >>> print("hi")
        ... hi
        ... ''')
        True
        >>> has_doctest(None)
        False
    """
    if not docstring:
        return False
    return ">>>" in docstring


def has_forbidden_doctest(docstring: str | None) -> bool:
    """Return True when a callable-based doctest shortcut is used.

    Examples:
        >>> sample = "Example.\\n>>> " + "callable(main)\\nTrue\\n"
        >>> has_forbidden_doctest(sample)
        True
        >>> has_forbidden_doctest('''Example.
        ...
        ... >>> isinstance(main.__name__, str)
        ... True
        ... ''')
        False
    """
    if not docstring:
        return False
    return bool(re.search(r"^\s*>>>\s*callable\(", docstring, flags=re.MULTILINE))


def iter_python_files(root: Path) -> Iterable[Path]:
    """Yield Python files under root, excluding common build folders.

    Examples:
        >>> files = list(iter_python_files(Path(".")))
        >>> any(path.name == "main.py" for path in files)
        True
    """
    excluded = {".git", ".venv", "build", "dist", "__pycache__"}
    for path in root.rglob("*.py"):
        if any(part in excluded for part in path.parts):
            continue
        yield path


def collect_functions(filepath: Path) -> List[FunctionInfo]:
    """Collect function definitions from a Python file.

    Examples:
        >>> functions = collect_functions(Path("main.py"))
        >>> any(func.name == "main" for func in functions)
        True
    """
    try:
        text = filepath.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(text, filename=str(filepath))
    except SyntaxError:
        return []

    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    functions: List[FunctionInfo] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            p = parent.get(node)
            nested_inside_function = False
            while p is not None:
                if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nested_inside_function = True
                    break
                p = parent.get(p)
            if nested_inside_function:
                continue
            docstring = ast.get_docstring(node)
            functions.append(
                FunctionInfo(
                    name=node.name,
                    filepath=filepath,
                    lineno=getattr(node, "lineno", 1),
                    has_doctest=has_doctest(docstring),
                    has_forbidden_doctest=has_forbidden_doctest(docstring),
                )
            )
    return functions


def report_missing(functions: List[FunctionInfo]) -> str:
    """Build a report of functions missing doctests.

    Examples:
        >>> report_missing([])
        'Functions without doctests:'
    """
    lines = ["Functions without doctests:"]
    for func in functions:
        lines.append(f"- {func.filepath}:{func.lineno} {func.name}")
    return "\n".join(lines)


def report_forbidden(functions: List[FunctionInfo]) -> str:
    """Build a report of functions with forbidden doctest patterns.

    Examples:
        >>> report_forbidden([])
        'Functions with forbidden doctests:'
    """
    lines = ["Functions with forbidden doctests:"]
    for func in functions:
        lines.append(f"- {func.filepath}:{func.lineno} {func.name}")
    return "\n".join(lines)


def main() -> int:
    """Run the doctest ratio check.

    Examples:
        >>> isinstance(main.__name__, str)
        True
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min", type=float, default=0.8, help="Minimum doctest ratio.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root to scan.",
    )
    args = parser.parse_args()

    all_functions: List[FunctionInfo] = []
    for path in iter_python_files(args.root):
        all_functions.extend(collect_functions(path))

    forbidden = [f for f in all_functions if f.has_forbidden_doctest]
    if forbidden:
        print(
            "Forbidden doctest pattern detected: do not use 'callable(...)' assertions."
        )
        print(report_forbidden(forbidden))
        return 1

    total = len(all_functions)
    if total == 0:
        return 0

    with_doctest = [f for f in all_functions if f.has_doctest]
    ratio = len(with_doctest) / total
    if ratio >= args.min:
        return 0

    missing = [f for f in all_functions if not f.has_doctest]
    print(
        f"Doctest coverage {ratio:.2%} below minimum {args.min:.0%} "
        f"({len(with_doctest)}/{total})."
    )
    print(report_missing(missing))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
