"""General-purpose helpers shared across notebooks and modules."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path object."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

