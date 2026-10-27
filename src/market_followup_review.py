"""Review persisted original filings and observed trading gaps, without API calls."""

from datetime import datetime, timedelta, timezone
import json
import sys
import unittest

import pandas as pd

from .collection_runner import ProgressLog
from .config import PROJECT_ROOT, RAW_DATA_DIR, INTERIM_DATA_DIR
from .market_sample import market_number


def main() -> int:
    """Tee tests and source comparisons, preserving unresolved valuation semantics."""
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_%f")
    output = PROJECT_ROOT / "reports/monthly" / f"task03_market_followup_review_{stamp}.json"
    log = ProgressLog(output.with_suffix(".txt"))
    result = {"cases": []}
    try:
        tests = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(
            unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"), top_level_dir=str(PROJECT_ROOT)))
        result["tests"] = {"run": tests.testsRun, "passed": tests.wasSuccessful(),
            "failures": len(tests.failures), "errors": len(tests.errors)}
        if not tests.wasSuccessful():
            return 1
        report = json.loads((PROJECT_ROOT / "reports/monthly/task03_market_followup_20261028_010029_898544.json").read_text(encoding="utf-8"))
        manifest = json.loads((INTERIM_DATA_DIR / "market_pilot_manifest_20261028_004741_026994.json").read_text(encoding="utf-8"))
        snapshots = pd.read_csv(RAW_DATA_DIR / "market_pilot_snapshots_20261028_004741_026994.csv", dtype="string")
        histories = pd.read_csv(PROJECT_ROOT / report["history_path"], dtype="string")
        for item, case in zip(report["cases"], manifest, strict=True):
            anchor = pd.Timestamp(case["observation_date_candidate"])
            matching = snapshots.loc[snapshots.ISU_SRT_CD.eq(case["stock_code"])
                & pd.to_datetime(snapshots.requested_trade_date).lt(anchor)].sort_values("requested_trade_date")
            current = matching.iloc[-1]
            last = histories.loc[histories.requested_isin.eq(current.ISU_CD)
                & pd.to_datetime(histories.TRD_DD).eq(pd.Timestamp(item["last_positive_volume_date_in_covered_history"]))]
            last_prices = set(last.TDD_CLSPRC.map(market_number))
            review = json.loads((PROJECT_ROOT / item["review_material"]).read_text(encoding="utf-8"))
            tables = [table for table in review["stock_tables"] if table["document"] == str(case["rcept_no"]) + ".xml"
                and "주식의 종류" in "".join(table["rows"][0])]
            entry = {"case": item["case"], "trading_gap_calendar_days": item["calendar_days_before_T"],
                "history_reaches_snapshot_day": item["history_reaches_snapshot_day"],
                "reported_snapshot_equals_last_observed_history_close": len(last_prices) == 1 and market_number(current.TDD_CLSPRC) in last_prices,
                "price_adjustment_contract_verified": False,
                "stock_class_table_located": len(tables) == 1,
                "T_date_full_equity_verified": False}
            if len(tables) == 1:
                table = tables[0]
                issued = [row for row in table["rows"] if row and row[0].startswith("Ⅳ.")]
                if len(issued) == 1 and len(issued[0]) >= 4:
                    row = issued[0]
                    entry["reported_common_equals_reported_total"] = market_number(row[1]) is not None and market_number(row[1]) == market_number(row[3])
                    entry["other_class_cell_is_dash"] = row[2] == "-"
                    entry["fiscal_report_common_matches_snapshot_listed_shares"] = market_number(row[1]) is not None and market_number(row[1]) == market_number(current.LIST_SHRS)
                    entry["stock_table_reference"] = {"document": table["document"], "table_index": table["table_index"]}
            # A manual source assessment below is kept separate from string matching.
            entry["suspension_paragraph_count"] = len(review["suspension_paragraphs"])
            result["cases"].append(entry)
        result["review_execution_passed"] = True
        result["valuation_approved"] = False
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
