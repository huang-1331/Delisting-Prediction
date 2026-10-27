"""Verify persisted smoke outputs and restart equality without API calls."""

import argparse
from datetime import datetime, timedelta, timezone
import json

import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import RequestStore
from .config import PROJECT_ROOT, COLLECTION_STATE_PATH


def main() -> int:
    """Read a successful smoke report and tee exact persisted-output comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    args = parser.parse_args()
    report = json.loads((PROJECT_ROOT / args.report).read_text(encoding="utf-8"))
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_%f")
    output = PROJECT_ROOT / "reports/monthly" / f"task02_review_{stamp}.json"
    log = ProgressLog(output.with_suffix(".txt"))
    result = {"source_report": args.report, "checks": {}}
    try:
        checks = result["checks"]
        checks["smoke_passed"] = bool(report["passed"] and report["live_verified"])
        checks["original_sources_unchanged"] = bool(report["original_sources_unchanged"])
        checks["historical_firms_included"] = report["candidates"]["historical_ready_linked"] > 0
        checks["unknown_industry_not_confirmed"] = report["candidates"]["industry_confirmed"] == 0
        for stage, batches in report["live_batches"].items():
            first = pd.read_csv(PROJECT_ROOT / batches["first"]["output"], dtype="string")
            restart = pd.read_csv(PROJECT_ROOT / batches["restart"]["output"], dtype="string")
            checks[f"{stage}_restart_exact_equal"] = first.equals(restart)
            checks[f"{stage}_restart_no_requests"] = batches["restart"]["requests"] == 0
            checks[f"{stage}_no_exact_duplicate_rows"] = not bool(first.duplicated().any())
            if stage == "statements":
                checks["snapshots_not_claimed_point_in_time"] = first.point_in_time_verified.eq("False").all().item()
        store = RequestStore(COLLECTION_STATE_PATH)
        try:
            counts = dict(store.db.execute("SELECT status,COUNT(*) FROM requests GROUP BY status"))
            result["persisted_outcomes"] = counts
            checks["persisted_request_keys_unique"] = store.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == store.db.execute("SELECT COUNT(*) FROM (SELECT DISTINCT stage,request_key FROM requests)").fetchone()[0]
            checks["no_legacy_status_imported"] = store.db.execute("SELECT COUNT(*) FROM requests WHERE provenance != 'live_request'").fetchone()[0] == 0
        finally:
            store.close()
        result["passed"] = all(checks.values())
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result["passed"] else 1
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
