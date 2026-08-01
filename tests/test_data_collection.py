"""Tests for official-source collection and point-in-time universe logic."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data_collection import (
    STANDARD_LISTING_COLUMNS,
    DartApiError,
    attach_disclosure_receipt_dates,
    build_kosdaq_manufacturing_history,
    fetch_annual_statement_with_fallback,
    filter_observations_to_listing_window,
    link_listing_history_to_dart_codes,
)


def _listing_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "stock_code": "000001",
        "company_name": "제조기업",
        "market": "코스닥",
        "industry": "전자부품 제조업",
        "listing_date": pd.Timestamp("2016-01-01"),
        "delisting_date": pd.NaT,
        "delisting_reason": pd.NA,
        "security_type": "주권",
        "stock_type": "보통주",
        "source": "test",
        "collected_at_utc": "2026-08-01T00:00:00+00:00",
    }
    row.update(overrides)
    return row


class FakeDartClient:
    """Return no CFS data and one OFS row for fallback testing."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch_financial_statement(
        self,
        corp_code: str,
        business_year: int,
        *,
        report_code: str,
        fs_div: str,
    ) -> pd.DataFrame:
        self.calls.append(fs_div)
        if fs_div == "CFS":
            raise DartApiError("013", "No data")
        return pd.DataFrame(
            {
                "corp_code": [corp_code],
                "bsns_year": [str(business_year)],
                "fs_div_used": [fs_div],
            }
        )


class DataCollectionTests(unittest.TestCase):
    """Verify universe construction without live network calls."""

    def test_annual_statement_falls_back_to_ofs(self) -> None:
        client = FakeDartClient()

        result = fetch_annual_statement_with_fallback(client, "00123456", 2024)

        self.assertEqual(client.calls, ["CFS", "OFS"])
        self.assertEqual(result.loc[0, "fs_div_used"], "OFS")

    def test_history_keeps_only_kosdaq_manufacturing_common_equity(self) -> None:
        current = pd.DataFrame(
            [
                _listing_row(stock_code="1"),
                _listing_row(stock_code="2", industry="소프트웨어 개발업"),
            ],
            columns=STANDARD_LISTING_COLUMNS,
        )
        delisted = pd.DataFrame(
            [
                _listing_row(
                    stock_code="3",
                    delisting_date=pd.Timestamp("2020-06-01"),
                    delisting_reason="감사의견 거절",
                ),
                _listing_row(stock_code="4", market="유가증권"),
                _listing_row(stock_code="5", stock_type="우선주"),
            ],
            columns=STANDARD_LISTING_COLUMNS,
        )

        result = build_kosdaq_manufacturing_history(current, delisted)

        self.assertEqual(set(result["stock_code"]), {"1", "3"})

    def test_observation_must_fall_inside_listing_window(self) -> None:
        history = pd.DataFrame(
            [
                _listing_row(
                    stock_code="000001",
                    listing_date=pd.Timestamp("2018-01-01"),
                    delisting_date=pd.Timestamp("2021-01-01"),
                )
            ],
            columns=STANDARD_LISTING_COLUMNS,
        )
        observations = pd.DataFrame(
            {
                "stock_code": ["000001", "000001", "000001"],
                "observation_date": ["2017-12-31", "2020-03-31", "2021-01-01"],
                "value": [1, 2, 3],
            }
        )

        result = filter_observations_to_listing_window(observations, history)

        self.assertEqual(result["value"].tolist(), [2])

    def test_dart_linkage_uses_stock_code_then_exact_name(self) -> None:
        history = pd.DataFrame(
            [
                _listing_row(stock_code="1", company_name="현재기업"),
                _listing_row(stock_code="9", company_name="(주)과거 기업"),
                _listing_row(stock_code="8", company_name="미해결기업"),
            ],
            columns=STANDARD_LISTING_COLUMNS,
        )
        dart_codes = pd.DataFrame(
            {
                "corp_code": ["C1", "C2"],
                "stock_code": ["000001", ""],
                "company_name": ["현재기업", "과거기업"],
            }
        )

        result = link_listing_history_to_dart_codes(history, dart_codes)

        methods = result.linked.set_index("corp_code")["corp_code_match_method"].to_dict()
        self.assertEqual(methods, {"C1": "stock_code", "C2": "company_name_exact"})
        self.assertEqual(result.unresolved["company_name"].tolist(), ["미해결기업"])

    def test_receipt_date_is_used_as_observation_time(self) -> None:
        statements = pd.DataFrame(
            {"rcept_no": ["202503310001", "202503310001"], "account_nm": ["자산", "부채"]}
        )
        disclosures = pd.DataFrame(
            {"rcept_no": ["202503310001"], "rcept_dt": ["2025-03-31"]}
        )

        result = attach_disclosure_receipt_dates(statements, disclosures)

        self.assertTrue(result["observation_date"].eq(pd.Timestamp("2025-03-31")).all())


if __name__ == "__main__":
    unittest.main()
