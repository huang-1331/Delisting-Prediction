"""Tee actual offline and optional bounded live collection smoke-test execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import json
import sys
import unittest

import pandas as pd

from .collection_runner import ProgressLog, collect_universe
from .collection_state import RequestStore, file_sha256
from .config import PROJECT_ROOT, RAW_DATA_DIR, INTERIM_DATA_DIR, COLLECTION_STATE_PATH
from .reliable_collection import run_batch


def main() -> int:
    """Validate real local history and optionally request two delisted companies.

    Output includes aggregates and artifact paths only. Live raw snapshots stay
    outside final model data. Existing source files are verified unchanged.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task02_smoke_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    result = {"at_kst": now.isoformat(), "live_requested": args.live,
              "log": report.with_suffix(".txt").relative_to(PROJECT_ROOT).as_posix()}
    original_files = [*RAW_DATA_DIR.glob("dart_*.csv"), *RAW_DATA_DIR.glob("kind_*.csv"),
                      *RAW_DATA_DIR.glob("krx_*.csv"), *INTERIM_DATA_DIR.glob("dart_*checkpoint*.json")]
    before = {path: file_sha256(path) for path in original_files}
    exit_code = 1
    try:
        print(f"Task02 actual smoke execution: {now.isoformat()}; live={args.live}", flush=True)
        tests = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(
            unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"), top_level_dir=str(PROJECT_ROOT)))
        result["tests"] = {"run": tests.testsRun, "errors": len(tests.errors),
                           "failures": len(tests.failures), "passed": tests.wasSuccessful()}
        if not tests.wasSuccessful():
            return 1
        paths = collect_universe(date(2026, 10, 28), log, source_date=date(2026, 8, 1))
        result["candidate_artifacts"] = {name: path.relative_to(PROJECT_ROOT).as_posix() for name, path in paths.items()}
        candidates = pd.read_csv(paths["candidates"], dtype="string")
        linked = pd.read_csv(paths["linked"], dtype="string")
        ready = linked.loc[linked.collection_status.eq("ready")]
        historical = ready.loc[ready.delisting_date.notna()].sort_values("stock_code")
        assert len(historical) > 0
        assert candidates.manufacturing_confirmed.isna().all()
        result["candidates"] = {"all_rows": len(candidates), "ready_linked": len(ready),
            "historical_ready_linked": len(historical), "industry_confirmed": 0,
            "identity_review": int(candidates.collection_status.eq("identity_review").sum()),
            "excluded_preserved": int(candidates.collection_status.eq("excluded").sum())}
        store = RequestStore(COLLECTION_STATE_PATH)
        try:
            store.ensure_recollection()
            result["collection_policy"] = "clean_recollection_no_legacy"
            result["legacy_imported_requests"] = store.db.execute("SELECT COUNT(*) FROM requests WHERE provenance != 'live_request'").fetchone()[0]
            result["state_outcomes_before_smoke"] = dict(store.db.execute("SELECT status,COUNT(*) FROM requests WHERE stage='statements' GROUP BY status"))
        finally:
            store.close()
        if args.live:
            # Technical smoke coverage, not a representative research sample.
            eligible_year = historical.loc[(pd.to_datetime(historical.listing_date) <= pd.Timestamp("2022-12-31"))
                & (pd.to_datetime(historical.delisting_date) >= pd.Timestamp("2023-01-01"))]
            sample = set(eligible_year.head(2).stock_code)
            if len(sample) != 2:
                raise ValueError("Two historical smoke candidates unavailable")
            result["live_sample"] = {"companies": 2, "annual_year": 2022,
                "selection": "first two stock-code-sorted identity-eligible historical listings covering FY2022; not an inference sample"}
            result["live_batches"] = {}
            for stage in ("statements", "disclosures"):
                first = run_batch(paths["linked"], date(2026, 10, 28), stage, log,
                                  years=[2022], stock_codes=sample, max_requests=2)
                second = run_batch(paths["linked"], date(2026, 10, 28), stage, log,
                                   years=[2022], stock_codes=sample, max_requests=2)
                record = {"first": asdict(first), "restart": asdict(second)}
                for part in record.values():
                    part["output"] = part["output"].relative_to(PROJECT_ROOT).as_posix()
                result["live_batches"][stage] = record
                if first.failed or second.failed or second.requests != 0 or first.rows != second.rows:
                    raise RuntimeError("Live request/restart smoke incomplete")
            result["live_verified"] = True
        else:
            result["live_verified"] = False
        exit_code = 0
    except Exception as exc:
        # No traceback or exception message may expose an authenticated URL.
        result["failure_type"] = type(exc).__name__
        print(f"Smoke incomplete: {type(exc).__name__}", flush=True)
    finally:
        result["original_sources_unchanged"] = all(file_sha256(path) == digest for path, digest in before.items())
        result["original_source_files_checked"] = len(before)
        if not result["original_sources_unchanged"]:
            exit_code = 1
        result["passed"] = exit_code == 0
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        log.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
