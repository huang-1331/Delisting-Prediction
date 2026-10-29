"""Protect taxonomy boundaries and historical reporting-entity evidence."""

import tempfile
from pathlib import Path
import unittest
from zipfile import ZipFile

from src.industry_evidence import (ksic_edition_at_T, manufacturing_divisions,
                                   parse_main_document, read_receipt, verify_public_date_page,
                                   business_evidence)
from src.config import INTERIM_DATA_DIR


class IndustryEvidenceTests(unittest.TestCase):
    def test_business_section_fallback_requires_entity_scope_review(self):
        source = ('<DOCUMENT><DOCUMENT-NAME>사업보고서</DOCUMENT-NAME>'
                  '<COMPANY-NAME AREGCIK="00111227">보고법인</COMPANY-NAME>'
                  '<SECTION-2><TITLE>1. 회사의 개요</TITLE><P>설립일 설명</P></SECTION-2>'
                  '<SECTION-2><TITLE>1. 사업의 개요</TITLE><P>모회사 영업중단; 자회사 제조업</P>'
                  '</SECTION-2></DOCUMENT>')
        tree, _ = parse_main_document(source.encode('utf-8'), '00111227')
        evidence = business_evidence(tree)
        self.assertEqual(len(evidence), 1)
        self.assertIn('requires_entity_scope_review', evidence[0]['selection'])
        self.assertNotIn('manufacturing_confirmed', evidence[0])

    def test_receipt_prefix_is_not_the_official_public_date(self):
        source = ('<html><head><title>코엔텍/사업보고서/2021.03.17</title></head>'
                  '<body><select><option value="rcpNo=20210316001238" selected>'
                  '2021.03.17 사업보고서</option></select></body></html>').encode('utf-8')
        result = verify_public_date_page(source, '20210316001238', '2021-03-17', '코엔텍')
        self.assertTrue(result['public_date_crosschecked'])
        with self.assertRaises(ValueError):
            verify_public_date_page(source, '20210316001238', '2021-03-16', '코엔텍')
        with self.assertRaises(ValueError):
            verify_public_date_page(source, '20210316001239', '2021-03-17', '코엔텍')

    def test_enactment_not_announcement_dates_and_changed_scope(self):
        for value, expected in [("2017-01-13", 9), ("2017-06-30", 9),
            ("2017-07-01", 10), ("2024-01-01", 10), ("2024-06-30", 10),
            ("2024-07-01", 11), ("2025-06-30", 11)]:
            self.assertEqual(ksic_edition_at_T(value), expected)
        self.assertNotIn(34, manufacturing_divisions(9))
        self.assertIn(34, manufacturing_divisions(10))
        self.assertIn(34, manufacturing_divisions(11))
        with self.assertRaises(ValueError):
            ksic_edition_at_T("2026-08-02")

    def test_cp949_wrong_declaration_and_leading_slash(self):
        source = '''<?xml version="1.0" encoding="utf-8"?>
        <DOCUMENT><DOCUMENT-NAME>사업보고서</DOCUMENT-NAME>
        <DOCUMENT-HEADER><COMPANY-NAME AREGCIK="00122205">보고법인</COMPANY-NAME></DOCUMENT-HEADER>
        <SECTION-1><SECTION-2><TITLE>1. 회사의 개요</TITLE><LIBRARY>
        <P>바. 주요사업의 내용</P><P>당사의 주된 사업은 외식업입니다.</P>
        <P>사. 계열회사</P><P>자회사의 제조업 설명</P></LIBRARY></SECTION-2></SECTION-1>
        <SECTION-1><P>주요사업 내용은 제조업인 자회사입니다.</P></SECTION-1></DOCUMENT>'''
        with tempfile.TemporaryDirectory(dir=INTERIM_DATA_DIR) as directory:
            path = Path(directory) / "source.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("/20150629000524.xml", source.encode("cp949"))
                archive.writestr("/20150629000524_00761.xml", b"attachment")
            result = read_receipt(path, "20150629000524", "00122205")
            self.assertEqual(result["decoded_encoding"], "cp949")
            self.assertEqual(result["declared_encoding"], "utf-8")
            text = " ".join(item["text"] for item in result["business_passages"])
            self.assertIn("주된 사업은 외식업", text)
            self.assertNotIn("자회사", text)
            with self.assertRaises(ValueError):
                read_receipt(path, "20150629000524", "99999999")

    def test_ambiguous_main_member_is_not_guessed(self):
        with tempfile.TemporaryDirectory(dir=INTERIM_DATA_DIR) as directory:
            path = Path(directory) / "source.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("/20150629000524.xml", b"one")
                archive.writestr("20150629000524.xml", b"two")
            with self.assertRaises(ValueError):
                read_receipt(path, "20150629000524", "00122205")

    def test_direct_document_identity_header_is_supported(self):
        source = ('<DOCUMENT><DOCUMENT-NAME>사업보고서</DOCUMENT-NAME>'
                  '<COMPANY-NAME AREGCIK="00111227">보고법인</COMPANY-NAME></DOCUMENT>')
        _, metadata = parse_main_document(source.encode("utf-8"), "00111227")
        self.assertEqual(metadata["reporting_name_at_T"], "보고법인")
        with self.assertRaises(ValueError):
            parse_main_document(source.encode("utf-8"), "99999999")


if __name__ == "__main__":
    unittest.main()
