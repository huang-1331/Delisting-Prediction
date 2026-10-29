"""Protect physical request limits and original/changed filing evidence."""

from io import BytesIO
import unittest
from unittest.mock import patch

import pandas as pd

from src.annual_receipt_inventory import annual_versions
from src.data_collection import DartApiClient, DartApiError
from src.task03_collect import PacedOpener


class Task03CollectionTests(unittest.TestCase):
    """Verify source limits and preserve unresolved source semantics."""

    def test_document_xml_rate_limit_preserves_documented_status(self):
        """A document API limit must stop work, not masquerade as a corrupt ZIP."""
        with patch("src.data_collection._read_url_bytes", return_value=b"<result><status>020</status><message>limit</message></result>"):
            with self.assertRaises(DartApiError) as raised:
                DartApiClient("test-only").fetch_document_archive("20230321000689")
        self.assertEqual(raised.exception.status, "020")

    def test_physical_request_budget_prevents_next_call(self):
        """Pagination and fallback calls consume the same physical budget."""
        opener = PacedOpener(1, interval=0)
        with patch("src.task03_collect.urlopen", return_value=BytesIO(b"ok")) as network:
            opener.open("test-only")
            with self.assertRaises(RuntimeError):
                opener.open("test-only")
            self.assertEqual(network.call_count, 1)
        self.assertTrue(opener.budget_exhausted)

    def test_inventory_preserves_amendments_non_december_and_unknowns(self):
        """Receipt titles cannot silently prove initial values or fiscal-year policy."""
        frame = pd.DataFrame({"report_nm": ["사업보고서 (2022.12)",
            "[기재정정]사업보고서 (2022.12)", "사업보고서 (2022.03)",
            "사업보고서 제출기한 연장신고서", "사업보고서 (2014.12)"]})
        selected, review = annual_versions(frame)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected.version_kind.tolist(),
            ["initial_candidate", "amendment_or_annotation", "initial_candidate"])
        self.assertEqual(selected.fiscal_month_reported.tolist(), [12, 12, 3])
        self.assertFalse(selected.first_version_verified.any())
        self.assertEqual(len(review), 1)


if __name__ == "__main__":
    unittest.main()
