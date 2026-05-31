"""Project-wide configuration.

This module centralizes paths, environment variables, and reproducibility
settings. Secrets such as the OpenDART API key must be supplied through
environment variables and must never be hardcoded in source files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RANDOM_STATE = 42
TARGET_COLUMN = "상장폐지_t12"
DART_API_KEY_ENV = "DART_API_KEY"


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    dart_api_key: str | None
    random_state: int = RANDOM_STATE
    target_column: str = TARGET_COLUMN


def get_dart_api_key(required: bool = True) -> str | None:
    """Return the OpenDART API key from the environment.

    Args:
        required: If True, raise an error when the key is missing.

    Returns:
        The API key string, or None when missing and not required.

    Raises:
        RuntimeError: If the key is required but not configured.
    """

    api_key = os.getenv(DART_API_KEY_ENV)
    if required and not api_key:
        raise RuntimeError(
            f"{DART_API_KEY_ENV} environment variable is required. "
            "Set it before running OpenDART collection code."
        )
    return api_key


def load_settings(required_dart_key: bool = False) -> Settings:
    """Load project settings from the current environment."""

    return Settings(dart_api_key=get_dart_api_key(required=required_dart_key))

