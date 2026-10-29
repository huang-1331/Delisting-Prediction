"""Project-wide configuration.

This module centralizes paths, local credentials, and reproducibility settings.
Credentials are loaded from the Git-ignored ``.secrets.toml`` file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
LOCAL_SECRETS_PATH = PROJECT_ROOT / ".secrets.toml"

RANDOM_STATE = 42
TARGET_COLUMN = "상장폐지_t12"
STUDY_START_YEAR = 2015
STUDY_END_YEAR = 2024

# User: use the KSIC edition generally effective at prediction T.
# Sources and the separate individual-law transition are in docs/ksic_at_T.md.
KSIC_EDITION_STARTS = (("2008-02-01", 9), ("2017-07-01", 10), ("2024-07-01", 11))
KSIC_MANUFACTURING_DIVISIONS = {9: (10, 33), 10: (10, 34), 11: (10, 34)}
# User choice A on 2026-10-28; source completeness must be verified separately.
EVENT_OBSERVATION_END = "2026-08-01"
ANNUAL_REPORT_CODE = "11011"
COLLECTION_STATE_PATH = INTERIM_DATA_DIR / "collection_recollection_v2.sqlite3"
COLLECTION_MAX_ATTEMPTS = 3
COLLECTION_RETRY_SECONDS = 1.0
MARKET_PROBE_STATE_PATH = INTERIM_DATA_DIR / "market_evidence_v1.sqlite3"

# Research decision only: the existing src/altman.py still implements original Z.
# TODO: implement Z-prime in a separately authorized modeling step.
ALTMAN_VARIANT = "z_prime"
ALTMAN_DISTRESS_CUTOFF = 1.23
ALTMAN_GREY_UPPER_CUTOFF = 2.90


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from the local secrets file."""

    dart_api_key: str | None = field(repr=False)
    krx_id: str | None = field(default=None, repr=False)
    krx_password: str | None = field(default=None, repr=False)
    random_state: int = RANDOM_STATE
    target_column: str = TARGET_COLUMN


def _load_local_secrets() -> dict[str, Any]:
    """Load credentials from the Git-ignored local TOML file."""

    if not LOCAL_SECRETS_PATH.exists():
        return {}

    try:
        with LOCAL_SECRETS_PATH.open("rb") as secrets_file:
            return tomllib.load(secrets_file)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(
            f"Invalid TOML syntax in {LOCAL_SECRETS_PATH.name}."
        ) from exc


def get_dart_api_key(required: bool = True) -> str | None:
    """Return the OpenDART API key from the local secrets file.

    Args:
        required: If True, raise an error when the key is missing.

    Returns:
        The API key string, or None when missing and not required.

    Raises:
        RuntimeError: If the key is required but not configured.
    """

    api_key = _load_local_secrets().get("opendart", {}).get("api_key")
    if required and not api_key:
        raise RuntimeError(
            f"OpenDART API key is required in {LOCAL_SECRETS_PATH.name}."
        )
    return api_key


def load_settings(required_dart_key: bool = False) -> Settings:
    """Load project settings from the local secrets file."""

    secrets = _load_local_secrets()
    krx = secrets.get("krx", {})

    return Settings(
        dart_api_key=get_dart_api_key(required=required_dart_key),
        krx_id=krx.get("id"),
        krx_password=krx.get("password"),
    )


def get_krx_credentials(required: bool = True) -> tuple[str | None, str | None]:
    """Return KRX credentials from the local secrets file.

    Args:
        required: If True, raise an error when either credential is missing.

    Returns:
        A tuple containing the KRX login ID and password.

    Raises:
        RuntimeError: If credentials are required but not configured.
    """

    krx = _load_local_secrets().get("krx", {})
    login_id = krx.get("id")
    password = krx.get("password")
    if required and not (login_id and password):
        raise RuntimeError(
            f"KRX ID and password are required in {LOCAL_SECRETS_PATH.name}."
        )
    return login_id, password

