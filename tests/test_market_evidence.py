"""Protect source semantics and original filing extraction for the market pilot."""

from datetime import date
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import pandas as pd

from src.config import INTERIM_DATA_DIR
from src.data_collection import DartApiClient, KrxDataClient
from src.market_followup import filing_review_material
from src.market_sample import annual_anchor, market_number


class MarketEvidenceTests(unittest.TestCase):
    """Exercise missing values, corrections, nonstandard XML cells and source contracts."""

    def test_numbers_do_not_turn_missing_or_nonfinite_values_into_zero(self):
        """Explicit zero remains zero; missing/invalid inputs remain None."""
        for value in (None, pd.NA, "-", "", "NaN", "Infinity"):
            self.assertIsNone(market_number(value))
        self.assertEqual(market_number("0"), Decimal(0))
        self.assertEqual(market_number("1,234"), Decimal(1234))

    def test_annual_anchor_excludes_corrections_and_other_periods(self):
        """A late correction or unrelated fiscal period cannot silently move T."""
        frame = pd.DataFrame({"report_nm": ["[기재정정]사업보고서 (2022.12)",
            "사업보고서 (2022.12)", "사업보고서 (2022.12)", "사업보고서 (2023.12)"],
            "rcept_dt": ["2023-03-01", "2023-03-22", "2023-03-21", "2024-03-20"],
            "rcept_no": ["A", "B", "C", "D"]})
        self.assertEqual(annual_anchor(frame, 2022).rcept_no, "C")
        self.assertIsNone(annual_anchor(frame, 2019))

    def test_dart_numeric_te_cells_and_preferred_dash_are_retained(self):
        """DART TE numeric cells must not disappear or become fabricated preferred shares."""
        with tempfile.TemporaryDirectory(prefix="market_test_", dir=INTERIM_DATA_DIR) as directory:
            assert Path(directory).resolve().is_relative_to(INTERIM_DATA_DIR.resolve())
            path = Path(directory) / "receipt.zip"
            source = '<html><head><meta charset="utf-8"></head><body><table><tr><td>발행할 주식의 총수</td><TE>1,000</TE><TE>-</TE></tr></table><p>회사는 매매거래정지 중입니다.</p></body></html>'
            with ZipFile(path, "w") as archive:
                archive.writestr("receipt.xml", source.encode("utf-8"))
            material = filing_review_material(path)
            self.assertEqual(material["stock_tables"][0]["rows"][0][1:], ["1,000", "-"])
            self.assertIn("te", material["stock_tables"][0]["cell_tags"])

    def test_history_response_keeps_unverified_price_adjustment_status(self):
        """Raw history does not silently become an unadjusted valuation input."""
        client = KrxDataClient("test-only", "test-only")
        client._authenticated = True
        client._request_json = lambda *args, **kwargs: {"output": [{"TRD_DD": "2023/03/20",
            "ACC_TRDVOL": "0", "TDD_CLSPRC": "1,000"}]}
        frame = client.fetch_price_history("test-isin", date(2023, 1, 1), date(2023, 3, 20))
        self.assertEqual(frame.loc[0, "price_adjustment_status"], "source_default_unverified")
        self.assertEqual(frame.loc[0, "ACC_TRDVOL"], "0")

    def test_original_archive_rejects_invalid_receipt_before_network(self):
        """No original document can be claimed for an invented/invalid receipt ID."""
        client = DartApiClient("test-only")
        with self.assertRaises(ValueError):
            client.fetch_document_archive("not-a-receipt")


if __name__ == "__main__":
    unittest.main()
