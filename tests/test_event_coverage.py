"""Regression checks for native/CSV event comparisons without inferred values."""

import unittest

import pandas as pd

from src.event_coverage_review import canonical_events


class EventComparisonTests(unittest.TestCase):
    """Check serialization differences and preserve substantive disagreements."""

    def fixture(self) -> pd.DataFrame:
        """Return one technical event fixture, not an actual research record."""
        return pd.DataFrame([{"stock_code": "000001", "company_name": "fixture",
            "market": "코스닥", "industry": "", "listing_date": "2010-01-01",
            "delisting_date": "2020-01-01", "delisting_reason": "",
            "security_type": "주권", "stock_type": "보통주"}])

    def test_native_blank_equals_csv_missing_without_mutating_source(self):
        """Empty source text and CSV NA compare equally while originals survive."""
        native = self.fixture()
        csv = native.replace("", pd.NA)
        pd.testing.assert_frame_equal(canonical_events(native), canonical_events(csv))
        self.assertEqual(native.loc[0, "industry"], "")

    def test_zero_and_substantive_reason_are_not_missing(self):
        """No numeric zero or reason can be normalized to an absent value."""
        native = self.fixture()
        changed = native.copy()
        changed.loc[0, "industry"] = "0"
        changed.loc[0, "delisting_reason"] = "합병"
        self.assertFalse(canonical_events(native).equals(canonical_events(changed)))
        self.assertEqual(canonical_events(changed).loc[0, "industry"], "0")


if __name__ == "__main__":
    unittest.main()
