"""Protect native periods, scopes, missing cells and separate recovery state."""

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.collection_state import RequestStore, statement_key, file_sha256
from src.config import INTERIM_DATA_DIR, PROJECT_ROOT
from src.statement_recovery import extract_source_tables, native_number
from src.reliable_collection import run_batch
from tests.test_collection_state import NullLog


def fixture() -> bytes:
    """Build explicit source-like statements, including TE cells and an equity grid."""
    groups = []
    for scope in ("C", "S"):
        for kind in ("BS", "IS", "CF", "EF"):
            caption = "제 24 기 2024.12.31 현재 제 23 기 2023.12.31 현재 제 22 기 2022.12.31 현재 (단위 : 원)"
            header = "<TR><TE></TE><TE>제 24 기</TE><TE>제 23 기</TE><TE>제 22 기</TE></TR>"
            if kind == "BS":
                rows = ''.join(f"<TR><TE>{label}</TE><TE>{amount}</TE><TE>{amount}</TE><TE>{amount}</TE></TR>" for label, amount in [("자산총계", "100"), ("부채총계", "60"), ("자본총계", "40")])
            elif kind == "EF":
                header = '<TR><TE colspan="3">자본</TE></TR>'
                rows = '<TR><TE>당기증감</TE><TE>-</TE><TE></TE></TR>'
            else:
                rows = '<TR><TE>원문항목</TE><TE>-</TE><TE>0</TE><TE>(1)</TE></TR>'
            groups.append(f'<TABLE-GROUP ACLASS="{{XBRL}}{kind}_{scope}"><TABLE><TR><TD>{caption}</TD></TR></TABLE><TABLE>{header}{rows}</TABLE></TABLE-GROUP>')
    return ('<?xml version="1.0" encoding="utf-8"?><DOCUMENT>' + ''.join(groups) + '</DOCUMENT>').encode('utf-8')


class StatementRecoveryTests(unittest.TestCase):
    """Test recovery invariants rather than recreating API output from later values."""

    def test_old_cp949_unsuffixed_tables_require_explicit_scope_heading(self):
        text = fixture().decode('utf-8').replace('_C', '')
        with self.assertRaises(ValueError):
            extract_source_tables(text.encode('cp949'), '20250324000028', 2024)
        text = text.replace('<DOCUMENT>', '<DOCUMENT><SECTION-2><TITLE>2. 연결재무제표</TITLE>')
        text = text.replace('</DOCUMENT>', '</SECTION-2></DOCUMENT>')
        recovered = extract_source_tables(text.encode('cp949'), '20250324000028', 2024)
        self.assertEqual(len(recovered['groups']), 8)
        self.assertEqual(len(recovered['balance_checks']), 6)

    def test_native_periods_missing_values_and_equity_grid_are_preserved(self):
        """Missing does not become zero; complex equity grids do not get guessed columns."""
        source = extract_source_tables(fixture(), "20250324000028", 2024)
        self.assertEqual(len(source["groups"]), 8)
        self.assertEqual(len(source["balance_checks"]), 6)
        amounts = [v["amount_source_text"] for v in source["period_values"] if v["statement_type"] == "IS"]
        self.assertIn("-", amounts)
        self.assertIn("0", amounts)
        self.assertIn("(1)", amounts)
        self.assertFalse(any(v["statement_type"] == "EF" for v in source["period_values"]))
        self.assertTrue(any(c["colspan"] == "3" for c in source["cells"]))
        self.assertIsNone(native_number("-"))
        self.assertEqual(native_number("(1)"), -1)

    def test_wrong_source_year_and_unbalanced_totals_are_rejected(self):
        """Source recovery cannot make a wrong year or inconsistent totals look verified."""
        with self.assertRaises(ValueError):
            extract_source_tables(fixture(), "20250324000028", 2023)
        altered = fixture().replace(b'<TE>100</TE>', b'<TE>101</TE>', 1)
        with self.assertRaises(ValueError):
            extract_source_tables(altered, "20250324000028", 2024)

    def test_restart_reuses_recovery_but_retains_api_failure_and_detects_tampering(self):
        """A verified alternate source prevents repeated bad calls without hiding failure."""
        with tempfile.TemporaryDirectory(prefix="recovery_test_", dir=INTERIM_DATA_DIR) as tmp:
            root = Path(tmp)
            self.assertTrue(root.resolve().is_relative_to(INTERIM_DATA_DIR.resolve()))
            payload = root / "source.xml"
            payload.write_bytes(fixture())
            parsed = extract_source_tables(fixture(), "20250324000028", 2024)
            key = statement_key("00123456", "000001", 2024)
            manifest = {"status": "verified_original_tables", "request_key": key,
                "business_year": 2024, "original_receipt": "20250324000028", "T": "2025-03-25",
                "files": [{"path": payload.relative_to(PROJECT_ROOT).as_posix(), "sha256": file_sha256(payload)}],
                "sources": [{"receipt": "20250324000028", "public_date": "2025-03-25", "groups": parsed["groups"]}],
                "balance_checks": parsed["balance_checks"]}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            state_path = root / "state.sqlite3"
            store = RequestStore(state_path)
            def failed():
                raise ValueError("Wrong source year")
            store.run("statements", key, failed)
            store.register_original_recovery(key, manifest_path)
            store.close()
            linked = root / "linked.csv"
            pd.DataFrame({"corp_code": ["00123456"], "stock_code": ["000001"],
                "collection_status": ["ready"], "corp_code_match_method": ["stock_code"]}).to_csv(linked, index=False)
            class NoNetwork:
                def fetch_financial_statement(inner, *args, **kwargs):
                    raise AssertionError("Verified original must be reused")
            result = run_batch(linked, date(2026, 8, 1), "statements", NullLog(),
                client=NoNetwork(), years=[2024], state_path=state_path, raw_dir=root)
            self.assertEqual((result.failed, result.recovered, result.requests, result.rows), (1, 1, 0, 0))
            store = RequestStore(state_path)
            try:
                self.assertEqual(store.status("statements", key), "failed")
                payload.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    store.original_recovery(key)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
