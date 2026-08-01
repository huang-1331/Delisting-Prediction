"""Tests for distress-event classification and T+12 labeling."""

from __future__ import annotations

import unittest

import pandas as pd

from src.config import TARGET_COLUMN
from src.labeling import (
    classify_delisting_reasons,
    create_t12_delisting_label,
    list_unresolved_delisting_reasons,
)


class LabelingTests(unittest.TestCase):
    """Verify target boundaries, censoring, and unresolved event handling."""

    def test_reason_classification_preserves_unknown_values(self) -> None:
        reasons = pd.Series(
            ["감사의견 거절", "피흡수합병", "기타 사유", pd.NA]
        )

        result = classify_delisting_reasons(reasons)

        self.assertTrue(bool(result.iloc[0]))
        self.assertFalse(bool(result.iloc[1]))
        self.assertTrue(pd.isna(result.iloc[2]))
        self.assertTrue(pd.isna(result.iloc[3]))

    def test_unresolved_reason_audit_counts_descriptions(self) -> None:
        events = pd.DataFrame(
            {"delisting_reason": ["기타 사유", "기타 사유", "감사의견 거절"]}
        )

        result = list_unresolved_delisting_reasons(events)

        self.assertEqual(result.to_dict("records"), [{"delisting_reason": "기타 사유", "count": 2}])

    def test_t12_labels_distress_and_censors_competing_events(self) -> None:
        observations = pd.DataFrame(
            {
                "stock_code": ["A", "B", "C", "D", "E"],
                "observation_date": [
                    "2024-01-31",
                    "2024-01-31",
                    "2024-01-31",
                    "2024-01-31",
                    "2024-06-01",
                ],
            }
        )
        events = pd.DataFrame(
            {
                "stock_code": ["A", "B", "C"],
                "delisting_date": ["2025-01-31", "2024-08-01", "2024-09-01"],
                "is_distress_delisting": pd.array([True, False, pd.NA], dtype="boolean"),
            }
        )

        result = create_t12_delisting_label(
            observations,
            events,
            label_as_of="2025-01-31",
        )

        expected = pd.array([1, pd.NA, pd.NA, 0, pd.NA], dtype="Int8")
        self.assertTrue(result[TARGET_COLUMN].array.equals(expected))
        self.assertNotIn("delisting_date", result.columns)
        self.assertNotIn("is_distress_delisting", result.columns)


if __name__ == "__main__":
    unittest.main()
