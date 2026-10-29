"""Verify that source encoding and prospective targets cannot alter evidence."""

import unittest

from src.spac_event_review import purpose_evidence


class SpacReviewTests(unittest.TestCase):
    """Protect review evidence without defining a new research target."""

    def test_xml_encoding_without_html_meta_is_honored(self):
        """DART XML declarations must work even when no HTML meta is present."""
        source = '<?xml version="1.0" encoding="utf-8"?><DOCUMENT><P>당사는 다른 회사와 합병하는 것을 유일한 사업목적으로 합니다.</P></DOCUMENT>'
        evidence = purpose_evidence(source.encode("utf-8"))
        self.assertEqual(len(evidence), 1)
        self.assertIn("당사는", evidence[0]["text"])
        self.assertTrue(evidence[0]["xpath"])

    def test_merger_target_manufacturing_is_not_parent_purpose_evidence(self):
        """A target-industry prospectus does not establish the parent company's activity."""
        source = '<html><head><meta charset="utf-8"></head><body><p>합병 대상 후보: 자동차 제조업 회사</p></body></html>'
        self.assertEqual(purpose_evidence(source.encode("utf-8")), [])

    def test_misdeclared_xml_does_not_force_invalid_utf8(self):
        """Actual HTML charset remains usable when the XML declaration is wrong."""
        source = '<?xml version="1.0" encoding="utf-8"?><html><head><meta charset="euc-kr"></head><body><p>당사는 다른 회사와 합병하는 것을 유일한 사업목적으로 합니다.</p></body></html>'
        evidence = purpose_evidence(source.encode("euc-kr"))
        self.assertEqual(len(evidence), 1)
        self.assertIn("당사는", evidence[0]["text"])


if __name__ == "__main__":
    unittest.main()
