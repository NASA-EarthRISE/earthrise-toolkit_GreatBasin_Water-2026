from __future__ import annotations

import json
import tokenize
from collections.abc import Iterator
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIRECTORIES = {
    ".git",
    ".quality-venv",
    ".venv",
    "venv",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "data",
    "past_project",
    "build",
    "dist",
    "node_modules",
}


def iter_active_files(pattern: str) -> Iterator[Path]:
    """
    Yield active repository files matching a glob pattern.

    Archived projects, data, environments, caches, and generated build
    directories are excluded from this initial assessment.
    """
    for path in ROOT.rglob(pattern):
        if not path.is_file():
            continue

        relative_path = path.relative_to(ROOT)

        if any(
            part in EXCLUDED_DIRECTORIES
            for part in relative_path.parts
        ):
            continue

        yield path


def test_readme_exists_and_is_not_empty() -> None:
    """The repository must contain usable top-level documentation."""
    readme = ROOT / "README.md"

    assert readme.is_file(), "README.md is missing."
    assert readme.stat().st_size > 0, "README.md exists but is empty."


def test_requirements_file_exists_and_is_not_empty() -> None:
    """The repository must declare its Python dependencies."""
    requirements = ROOT / "requirements.txt"

    assert requirements.is_file(), "requirements.txt is missing."
    assert requirements.stat().st_size > 0, (
        "requirements.txt exists but is empty."
    )


def test_repository_contains_active_python_files() -> None:
    """Confirm that the Python checks apply to this repository."""
    python_files = list(iter_active_files("*.py"))

    assert python_files, "No active Python files were found."


def test_active_python_files_can_be_parsed() -> None:
    """
    Parse Python files without importing or executing them.

    This detects syntax and encoding problems without triggering script
    side effects, data access, Earth Engine calls, or long computations.
    """
    failures: list[str] = []

    for path in iter_active_files("*.py"):
        try:
            with tokenize.open(path) as source_file:
                source = source_file.read()

            compile(source, str(path), "exec")

        except (SyntaxError, UnicodeError, OSError) as error:
            relative_path = path.relative_to(ROOT)
            failures.append(f"{relative_path}: {error}")

    assert not failures, (
        "The following Python files could not be parsed:\n"
        + "\n".join(failures)
    )


def test_active_notebooks_are_structurally_readable() -> None:
    """
    Confirm that active notebooks contain valid notebook-like JSON.

    Notebook cells are not executed and notebook files are not modified.
    """
    failures: list[str] = []

    for path in iter_active_files("*.ipynb"):
        relative_path = path.relative_to(ROOT)

        try:
            document = json.loads(
                path.read_text(encoding="utf-8-sig")
            )
        except (json.JSONDecodeError, UnicodeError, OSError) as error:
            failures.append(f"{relative_path}: {error}")
            continue

        if not isinstance(document, dict):
            failures.append(
                f"{relative_path}: notebook root is not an object"
            )
            continue

        if not isinstance(document.get("cells"), list):
            failures.append(
                f"{relative_path}: missing or invalid cells list"
            )

        if not isinstance(document.get("metadata"), dict):
            failures.append(
                f"{relative_path}: missing or invalid metadata object"
            )

        if not isinstance(document.get("nbformat"), int):
            failures.append(
                f"{relative_path}: missing or invalid nbformat value"
            )

    assert not failures, (
        "The following notebooks are structurally invalid:\n"
        + "\n".join(failures)
    )