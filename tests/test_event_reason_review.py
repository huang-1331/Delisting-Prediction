"""Keep event evidence review separate from ambiguous control-change phrases."""

import unittest

from src.event_reason_review import reviewed_reason_category


class EventReasonReviewTests(unittest.TestCase):
    def test_explicit_capital_reason_and_ambiguous_contexts(self):
        self.assertEqual(reviewed_reason_category("자본전액잠식"), "capital_impairment")
        for text in ["타법인의 완전자회사로 편입", "상장폐지 신청('22.08.31)",
                     "상장예비심사청구서 미제출", "합병 예비심사 미제출 관리종목"]:
            self.assertIsNone(reviewed_reason_category(text))


if __name__ == "__main__":
    unittest.main()
