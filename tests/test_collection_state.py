"""Meaningful restart, migration and candidate safety tests without credentials."""

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.collection_state import RequestStore, statement_key, file_sha256
from src.data_collection import build_collection_candidates, link_listing_history_to_dart_codes, DartApiError
from src.reliable_collection import run_batch
from src.config import INTERIM_DATA_DIR
from tests.test_data_collection import _listing_row


class NullLog:
    """Discard test-only progress; the smoke runner tees unittest's real output."""
    def line(self, message: str) -> None:
        """Accept a progress line without reading or writing credentials."""
        pass


class CollectionStateTests(unittest.TestCase):
    """Exercise failures around transactional storage, not just constructors."""

    def setUp(self) -> None:
        """Create an isolated store and source folder for each case."""
        INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="task02_test_", dir=INTERIM_DATA_DIR)
        self.root = Path(self.temporary.name)
        assert self.root.resolve().is_relative_to(INTERIM_DATA_DIR.resolve())
        self.raw = self.root / "raw"
        self.interim = self.root / "interim"
        self.raw.mkdir()
        self.interim.mkdir()
        self.store = RequestStore(self.interim / "state.sqlite3")

    def tearDown(self) -> None:
        """Close SQLite before removing the isolated test directory."""
        self.store.close()
        self.temporary.cleanup()

    def test_failed_request_retries_without_duplicate_success(self) -> None:
        """A transient failure is retryable; success is never re-fetched."""
        calls = []
        def fetch():
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("https://test.invalid/?crtfc_key=DO_NOT_STORE")
            return pd.DataFrame({"value": [1, 2]})
        self.assertEqual(self.store.run("x", "k", fetch)[0], "failed")
        self.assertEqual(self.store.run("x", "k", fetch)[0], "success")
        self.assertTrue(self.store.run("x", "k", fetch)[1])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.store.frame("x", "k")), 2)
        self.assertNotIn("DO_NOT_STORE", str(self.store.db.execute("SELECT * FROM requests").fetchall()))

    def test_unavailable_is_distinct_and_can_be_rechecked_explicitly(self) -> None:
        """No-data is reusable but a user-requested refresh may find new data."""
        self.assertEqual(self.store.run("x", "k", lambda: pd.DataFrame())[0], "unavailable")
        self.assertTrue(self.store.run("x", "k", lambda: self.fail("must not refetch"))[1])
        self.assertEqual(self.store.run("x", "k", lambda: pd.DataFrame({"x": [1]}), True)[0], "success")

    def test_process_interruption_rolls_back_and_restart_recovers(self) -> None:
        """A crash between response and commit leaves no false success."""
        original = self.store._put
        def interrupted(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt()
        self.store._put = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.store.run("x", "k", lambda: pd.DataFrame({"x": [1]}))
        self.store.close()
        self.store = RequestStore(self.interim / "state.sqlite3")
        self.assertIsNone(self.store.status("x", "k"))
        self.assertEqual(self.store.run("x", "k", lambda: pd.DataFrame({"x": [1]}))[0], "success")

    def test_legacy_data_issue_and_unproven_completion_are_reconciled(self) -> None:
        """Rows outside checkpoint survive; old failed completions are retryable."""
        frame = pd.DataFrame({"corp_code": ["00123456"], "stock_code": ["000001"],
            "business_year_requested": [2020], "rcept_no": ["20210301000001"], "fs_div_used": ["CFS"], "account_nm": ["자산"]})
        raw = self.raw / "dart_annual_statements_2015_2024_2026-08-01.csv"
        frame.to_csv(raw, index=False)
        issues = self.raw / "dart_annual_statement_issues_2026-08-01.csv"
        pd.DataFrame([{"corp_code": "00123456", "stock_code": "000001", "business_year": 2021, "issue": "No CFS or OFS annual statement returned"},
            {"corp_code": "00123456", "stock_code": "000001", "business_year": 2022, "issue": "Timeout"}]).to_csv(issues, index=False)
        checkpoint = self.interim / "dart_annual_statement_checkpoint_2026-08-01.json"
        checkpoint.write_text(json.dumps({"completed": ["00123456|000001|2022", "00123456|000001|2023"]}))
        before = {p: file_sha256(p) for p in (raw, issues, checkpoint)}
        result = self.store.migrate_legacy(self.raw, self.interim)
        self.assertEqual(result["success"], 1)
        self.assertEqual(self.store.status("statements", statement_key("00123456", "000001", 2020)), "success")
        self.assertEqual(self.store.status("statements", statement_key("00123456", "000001", 2021)), "unavailable")
        for year in (2022, 2023):
            self.assertEqual(self.store.status("statements", statement_key("00123456", "000001", year)), "failed")
        self.assertEqual(before, {p: file_sha256(p) for p in before})
        self.assertEqual(self.store.migrate_legacy(self.raw, self.interim)["files"], 0)

    def test_changed_export_date_and_filename_do_not_reset_success(self) -> None:
        """A stable request identity survives a newly named export and restart."""
        self.store.run("statements", "k", lambda: pd.DataFrame({"x": [1]}))
        self.store.export("statements", ["k", "k"], self.raw / "old.csv")
        self.store.close()
        self.store = RequestStore(self.interim / "state.sqlite3")
        self.assertTrue(self.store.run("statements", "k", lambda: self.fail("success must survive"))[1])
        self.store.export("statements", ["k"], self.raw / "new_date.csv")
        self.assertEqual(len(pd.read_csv(self.raw / "new_date.csv")), 1)
        self.assertTrue((self.raw / "old.csv").exists())

    def test_disclosure_date_range_changes_request_identity(self) -> None:
        """An earlier range's success cannot suppress a later range's request."""
        calls = []
        def fetch():
            calls.append(1)
            return pd.DataFrame({"x": [1]})
        self.store.run("disclosures", "C|S|2015-01-01|2026-08-01|A", fetch)
        self.store.run("disclosures", "C|S|2015-01-01|2026-10-28|A", fetch)
        self.assertEqual(len(calls), 2)

    def test_unrecognized_legacy_versions_are_not_declared_success(self) -> None:
        """Multiple receipt versions require review instead of arbitrary deduplication."""
        pd.DataFrame({"corp_code": ["00123456"] * 2, "stock_code": ["000001"] * 2,
            "business_year_requested": [2020] * 2, "rcept_no": ["old", "new"],
            "fs_div_used": ["CFS"] * 2}).to_csv(self.raw / "dart_annual_statements_conflict.csv", index=False)
        self.store.migrate_legacy(self.raw, self.interim)
        self.assertEqual(self.store.status("statements", statement_key("00123456", "000001", 2020)), "failed")

    def test_batch_retries_and_date_change_reuses_committed_data(self) -> None:
        """The collector integrates bounded retries, validation, export and restart."""
        path = self.interim / "linked.csv"
        pd.DataFrame({"corp_code": ["00123456"], "stock_code": ["000001"],
            "collection_status": ["ready"], "corp_code_match_method": ["stock_code"]}).to_csv(path, index=False)
        class Client:
            calls = 0
            def fetch_financial_statement(inner, corp, year, **kwargs):
                inner.calls += 1
                if inner.calls == 1:
                    raise TimeoutError()
                return pd.DataFrame({"corp_code": [corp], "bsns_year": [str(year)],
                    "reprt_code": ["11011"], "rcept_no": ["20210301000001"], "fs_div_used": ["CFS"]})
        client = Client()
        options = dict(client=client, years=[2020], raw_dir=self.raw, interim_dir=self.interim,
            state_path=self.interim / "batch.sqlite3", retry_seconds=0)
        first = run_batch(path, date(2026, 8, 1), "statements", NullLog(), **options)
        second = run_batch(path, date(2026, 10, 28), "statements", NullLog(), **options)
        self.assertEqual((first.success, first.requests, first.rows), (1, 2, 1))
        self.assertEqual((second.success, second.requests, second.skipped, second.rows), (1, 0, 1, 1))
        self.assertNotEqual(first.output, second.output)
        self.assertFalse(pd.read_csv(second.output).point_in_time_verified.any())

    def test_batch_rejects_wrong_company_response(self) -> None:
        """A successful HTTP response for another corporation is a failed request."""
        path = self.interim / "linked.csv"
        pd.DataFrame({"corp_code": ["00123456"], "stock_code": ["000001"],
            "collection_status": ["ready"], "corp_code_match_method": ["stock_code"]}).to_csv(path, index=False)
        class Client:
            def fetch_financial_statement(inner, corp, year, **kwargs):
                return pd.DataFrame({"corp_code": ["wrong"], "bsns_year": [str(year)],
                    "reprt_code": ["11011"], "rcept_no": ["20210301000001"], "fs_div_used": ["CFS"]})
        result = run_batch(path, date(2026, 10, 28), "statements", NullLog(), client=Client(),
            years=[2020], raw_dir=self.raw, interim_dir=self.interim,
            state_path=self.interim / "batch.sqlite3", max_attempts=1)
        self.assertEqual((result.failed, result.success, result.rows), (1, 0, 0))

    def test_choice_b_ignores_malformed_legacy_files_and_preserves_checkpoint(self) -> None:
        """Recollection never infers successes from corrupted old CSV or checkpoints."""
        legacy = self.raw / "dart_annual_statements_legacy.csv"
        legacy.write_text('a,b\n1,2\n3,4,5,6\n')
        checkpoint = self.interim / "dart_annual_statement_checkpoint_legacy.json"
        checkpoint.write_text(json.dumps({"completed": ["00123456|000001|2020"]}))
        before = {p: file_sha256(p) for p in (legacy, checkpoint)}
        path = self.interim / "linked.csv"
        pd.DataFrame({"corp_code": ["00123456"], "stock_code": ["000001"],
            "collection_status": ["ready"], "corp_code_match_method": ["stock_code"]}).to_csv(path, index=False)
        class Client:
            calls = 0
            def fetch_financial_statement(inner, corp, year, **kwargs):
                inner.calls += 1
                return pd.DataFrame({"corp_code": [corp], "bsns_year": [str(year)],
                    "reprt_code": ["11011"], "rcept_no": ["20210301000001"], "fs_div_used": ["CFS"]})
        client = Client()
        result = run_batch(path, date(2026, 10, 28), "statements", NullLog(), client=client,
            years=[2020], raw_dir=self.raw, interim_dir=self.interim,
            state_path=self.interim / "clean.sqlite3", max_attempts=1)
        self.assertEqual((result.success, client.calls, result.skipped), (1, 1, 0))
        self.assertEqual(before, {p: file_sha256(p) for p in before})
        clean = RequestStore(self.interim / "clean.sqlite3")
        try:
            with self.assertRaises(ValueError):
                clean.migrate_legacy(self.raw, self.interim)
        finally:
            clean.close()

    def test_wrong_year_response_is_failed_without_coercing_future_values(self) -> None:
        """A real observed API mismatch must never turn next-year data into this year."""
        path = self.interim / "linked.csv"
        pd.DataFrame({"corp_code": ["00123456"], "stock_code": ["000001"],
            "collection_status": ["ready"], "corp_code_match_method": ["stock_code"]}).to_csv(path, index=False)
        class Client:
            def fetch_financial_statement(inner, corp, year, **kwargs):
                return pd.DataFrame({"corp_code": [corp], "bsns_year": ["2025"],
                    "reprt_code": ["11011"], "rcept_no": ["20260324000091"],
                    "fs_div_used": ["CFS"], "thstrm_amount": ["123"]})
        state = self.interim / "wrong_year.sqlite3"
        result = run_batch(path, date(2026, 8, 1), "statements", NullLog(),
            client=Client(), years=[2024], state_path=state, raw_dir=self.raw,
            interim_dir=self.interim, max_attempts=1)
        self.assertEqual((result.failed, result.success, result.rows), (1, 0, 0))
        store = RequestStore(state)
        try:
            key = statement_key("00123456", "000001", 2024)
            self.assertEqual(store.status("statements", key), "failed")
            self.assertTrue(store.frame("statements", key).empty)
        finally:
            store.close()


class CandidateTests(unittest.TestCase):
    """Verify no industry or security identity is silently inferred."""

    def test_historical_unknown_industry_is_collected_but_not_model_ready(self) -> None:
        """A historical common-equity row survives without a manufacturing keyword."""
        rows = pd.DataFrame([_listing_row(industry="기계·장비", delisting_date="2020-01-01"),
                            _listing_row(stock_code="000002", stock_type=pd.NA)])
        result = build_collection_candidates(rows.iloc[:0], rows)
        self.assertEqual(result.collection_status.tolist(), ["ready", "identity_review"])
        self.assertTrue(result.manufacturing_confirmed.isna().all())
        self.assertFalse(result.final_sample_eligible.any())

    def test_non_equity_and_non_kosdaq_are_excluded_and_preserved(self) -> None:
        """Known ineligible records remain auditable, never requested."""
        rows = pd.DataFrame([_listing_row(security_type="신주인수권"),
                            _listing_row(stock_code="000002", market="유가증권")])
        result = build_collection_candidates(rows, rows.iloc[:0])
        self.assertEqual(result.collection_status.tolist(), ["excluded", "excluded"])

    def test_conflicting_security_identity_is_reviewed_not_overwritten(self) -> None:
        """Conflict rows survive without last-row-wins eligibility."""
        rows = pd.DataFrame([_listing_row(), _listing_row(stock_type="우선주")])
        result = build_collection_candidates(rows, rows.iloc[:0])
        self.assertEqual(result.collection_status.tolist(), ["identity_review"] * 2)

    def test_ambiguous_stock_code_cannot_fall_back_to_name(self) -> None:
        """One matching name must not override a conflicting stock identifier."""
        history = pd.DataFrame([_listing_row()])
        codes = pd.DataFrame({"corp_code": ["A", "B"], "stock_code": ["000001", "000001"],
                              "company_name": ["제조기업", "다른기업"]})
        linked = link_listing_history_to_dart_codes(history, codes)
        self.assertEqual(len(linked.linked), 0)
        self.assertEqual(len(linked.unresolved), 1)


if __name__ == "__main__":
    unittest.main()
