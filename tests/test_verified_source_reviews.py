"""Protect review counts from duplicated or unreviewed company-T facts."""

import unittest
import pandas as pd
from src.verified_source_reviews import combine_industry_reviews, explicit_full_year_dates


class ReviewedIndustryTests(unittest.TestCase):
    def test_source_dates_accept_single_digit_days_without_inventing_years(self):
        self.assertEqual(explicit_full_year_dates('주식교환일 2024년 11월 8일 2024.11.08'), {'2024-11-08'})
        self.assertEqual(explicit_full_year_dates('23.08.08'), set())

    def test_duplicate_T_and_unknown_are_not_claimed_reviewed(self):
        row = pd.DataFrame({'corp_code': ['00123456'], 'stock_code': ['000001'],
                            'T': ['2020-03-30'], 'manufacturing_confirmed': [1]})
        with self.assertRaises(ValueError):
            combine_industry_reviews(row, row.copy())
        unknown = row.assign(T='2021-03-30', manufacturing_confirmed=pd.NA)
        with self.assertRaises(ValueError):
            combine_industry_reviews(row, unknown)
        nonmanufacturing = row.assign(T='2021-03-30', manufacturing_confirmed=0)
        self.assertEqual(len(combine_industry_reviews(row, nonmanufacturing)), 2)


if __name__ == '__main__':
    unittest.main()
