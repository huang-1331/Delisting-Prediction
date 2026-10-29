"""Prevent periodic-report cache semantics from silently covering all types."""

from datetime import date
import unittest
from unittest.mock import Mock

from src.data_collection import DartApiClient


class EventContextTests(unittest.TestCase):
    def test_all_types_omits_filter_and_default_stays_periodic(self):
        client = DartApiClient("fixture")
        client._get_json = Mock(return_value={"list": [], "total_page": 1})
        client.fetch_disclosures("00123456", date(2015, 1, 1), date(2016, 1, 1))
        self.assertEqual(client._get_json.call_args.args[1]["pblntf_ty"], "A")
        client.fetch_disclosures("00123456", date(2015, 1, 1), date(2016, 1, 1), disclosure_type=None)
        self.assertNotIn("pblntf_ty", client._get_json.call_args.args[1])
        with self.assertRaises(ValueError):
            client.fetch_disclosures("00123456", date(2015, 1, 1), date(2016, 1, 1), disclosure_type="UNKNOWN")


if __name__ == "__main__":
    unittest.main()
