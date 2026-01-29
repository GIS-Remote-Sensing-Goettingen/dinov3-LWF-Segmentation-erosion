Here’s a **practical Python style guide inspired by the Google Python Style Guide** — but *adapted* to also emphasize **declarative, readable, maintainable code** that’s optimized for readers and team collaboration.

This combines:

* Google’s own Python style rules (formatting, docstrings, naming, etc.) ([Google GitHub][1])
* Core Python best practices (ability to read code *fast*) ([Real Python][2])
* Declarative, intention-expressing patterns where idiomatic Python makes sense (e.g., comprehensions, built-ins)

---

# Python Style Guide (Readable & Maintainable)

## 1) General Philosophy

**Code is read far more often than it is written.**
So style should help the *reader* immediately understand intent and logic. (Aligns with PEP 8 and Google style emphasis on readability). ([Python Enhancement Proposals (PEPs)][3])

---

## 2) Formatting

### Indentation & Whitespace

* Use **4 spaces per level** (standard Python). ([chromium.org][4])
* Avoid tabs.
* Wrap lines at ~79 characters (PEP 8 recommendation). ([Python Enhancement Proposals (PEPs)][3])

---

## 3) Naming Conventions

Good names tell the story of the code.

**Variables & Functions**

* Use **lower_case_with_underscores** for functions and variables. ([Python Enhancement Proposals (PEPs)][3])
* Choose descriptive names: `total_users`, not `x`.

**Classes**

* Use **CamelCase** for classes. ([Python Enhancement Proposals (PEPs)][3])

**Constants**

* Use **UPPER_SNAKE_CASE** for constants.

---

## 4) Docstrings & Comments

Every public function/class should be documented.

**Google-style docstring structure:**

```python
"""Short summary of what this does.

More detailed description if needed.

Args:
    arg1 (int): what it is.
    arg2 (str): what it is.

Returns:
    bool: what the return value means.
"""
```

Docstrings should explain *what* the function does — not *how* it does it. ([Chromium Git Repositories][5])

---

## 5) Declarative & Pythonic Patterns

**Goal:** Express *intent*, not step-by-step mechanics.

### 5a) List/Geneartor Expressions

Instead of loops that mutate:

```python
# Imperative
result = []
for user in users:
    if user.active:
        result.append(user.id)
```

Use declarative comprehension:

```python
# Declarative
result = [u.id for u in users if u.active]
```

This directly states:

> select `id` for every active user

---

### 5b) Built-ins over loops

Python built-ins express intent more concisely:

```python
# Use built-ins
total = sum(x * 10 for x in values if x > 0)
# rather than an explicit loop with mutation
```

---

### 5c) Declarative function composition

Prefer chaining built-ins instead of manual loops:

```python
result = sorted(
    set(values),
    key=lambda x: -x
)
```

**This declares: sort unique values descending** in a single high-level expression.

---

## 6) Functions Should Do One Thing

Encapsulate a single idea or purpose per function.

```python
def calculate_average(scores: list[float]) -> float:
    """Return the average of a list of scores."""
    return sum(scores) / len(scores)
```

Small, focused functions make code easier to understand and test.

---

## 7) Type Hints (Optional but Recommended)

Type hints help readers immediately know what types you expect.

```python
def find_max(users: list[User]) -> User | None:
    ...
```

They don’t change runtime behavior but *dramatically improve readability* and tooling support.

---

## 8) Error Handling

Use exceptions for truly exceptional cases, not control flow.

Prefer explicit checks:

```python
if not path.exists():
    raise FileNotFoundError("Config not found")
```

Over hidden errors.

---

## 9) Avoid Global State

Mutating globals makes understanding flow harder.

Instead, pass parameters and return results explicitly.

---
## 10)  avoid flags whe running a script
Prefer creating config.yml files  than using flags when running scripts exept specified.

---

---
## 11)  ALWAYS include docstrings and doctests
 And if necessary, add tests in the tests folder , for more compehensive and end to end tests.

---

## Example Putting It All Together

Below is a small example that adheres to all these principles.

```python
from __future__ import annotations

from pathlib import Path


def load_words(file_path: str) -> list[str]:
    """Load words from the given file, one per line.

    Lines are stripped and lowercased.

    Args:
        file_path (str): The path to the word file.

    Returns:
        list[str]: A list of cleaned words.

    Raises:
        FileNotFoundError: If file_path does not exist or is not a file.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "words.txt"
        ...     _ = p.write_text(" Hello\\nWORLD\\n\\n  test  \\n")
        ...     load_words(str(p))
        ['hello', 'world', '', 'test']
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # splitlines() preserves empty lines as "" entries.
    return [line.strip().lower() for line in path.read_text().splitlines()]


def filter_valid_words(words: list[str], min_length: int = 3) -> list[str]:
    """Return only the words with length >= min_length.

    Args:
        words (list[str]): Words to filter.
        min_length (int): Minimum allowed length of words.

    Returns:
        list[str]: Words of length >= min_length.

    Examples:
        >>> filter_valid_words(["a", "bee", "", "tree", "hi"], min_length=3)
        ['bee', 'tree']
        >>> filter_valid_words(["cat", "dog"], min_length=1)
        ['cat', 'dog']
    """
    return [w for w in words if len(w) >= min_length]


def unique_sorted(words: list[str]) -> list[str]:
    """Return words unique and sorted alphabetically.

    Examples:
        >>> unique_sorted(["b", "a", "b", "c"])
        ['a', 'b', 'c']
        >>> unique_sorted([])
        []
    """
    return sorted(set(words))


def process_word_list(file_path: str, min_length: int = 3) -> list[str]:
    """Load, filter, unique, and sort a word list.

    Args:
        file_path (str): Path to newline-separated words.
        min_length (int): Minimum allowed word length.

    Returns:
        list[str]: Unique, sorted words of length >= min_length.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "words.txt"
        ...     _ = p.write_text("Banana\\napple\\nApple\\nkiwi\\n")
        ...     process_word_list(str(p), min_length=5)
        ['apple', 'banana']
    """
    words = load_words(file_path)
    long_words = filter_valid_words(words, min_length=min_length)
    return unique_sorted(long_words)
```
