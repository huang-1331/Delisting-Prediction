"""OpenDART and market data collection utilities.

The functions in this module are intentionally small and testable. They should
only collect information observable at the requested point in time so that
downstream labeling and modeling do not accidentally introduce future data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from .config import get_dart_api_key
except ImportError:  # pragma: no cover - supports direct notebook imports
    from config import get_dart_api_key


@dataclass(frozen=True)
class DartConnectionResult:
    """Result object returned by OpenDART connection checks."""

    ok: bool
    message: str
    sample: Any | None = None


def create_dart_client(api_key: str | None = None) -> Any:
    """Create an OpenDartReader client.

    Args:
        api_key: Optional API key. If omitted, `DART_API_KEY` is read from the
            environment.

    Returns:
        An initialized OpenDartReader client.

    Raises:
        ImportError: If OpenDartReader is not installed.
        RuntimeError: If the API key is missing.
    """

    try:
        import OpenDartReader  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenDartReader is not installed. Install project dependencies "
            "with `pip install -r requirements.txt`."
        ) from exc

    resolved_api_key = api_key or get_dart_api_key(required=True)
    return OpenDartReader(resolved_api_key)


def test_dart_connection(company_name: str = "삼성전자") -> DartConnectionResult:
    """Test whether OpenDART can be reached with the configured API key.

    This function performs a lightweight lookup using a company name. The
    default company is only a connectivity smoke test and is not part of the
    KOSDAQ manufacturing research universe.

    Args:
        company_name: Company name used for a minimal lookup.

    Returns:
        A structured connection test result.
    """

    try:
        dart = create_dart_client()
        sample = dart.company_by_name(company_name)
    except Exception as exc:  # pragma: no cover - depends on external service
        return DartConnectionResult(
            ok=False,
            message=f"OpenDART connection test failed: {exc}",
            sample=None,
        )

    return DartConnectionResult(
        ok=True,
        message="OpenDART connection test succeeded.",
        sample=sample,
    )


def collect_financial_statements(*args: Any, **kwargs: Any) -> None:
    """Collect OpenDART financial statements.

    TODO:
        Define the exact KOSDAQ manufacturing company universe, statement
        type, period granularity, and storage schema before implementation.
    """

    raise NotImplementedError(
        "TODO: Implement after company universe and financial statement schema "
        "are finalized."
    )


if __name__ == "__main__":
    result = test_dart_connection()
    print(result.message)
    if result.sample is not None:
        print(result.sample)
