"""Inventory actual annual receipts and versions; never infer missing filings."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re

import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import RequestStore
from .config import (PROJECT_ROOT, INTERIM_DATA_DIR, COLLECTION_STATE_PATH,
                     EVENT_OBSERVATION_END, STUDY_START_YEAR, STUDY_END_YEAR)


def annual_versions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep every reported annual version, including non-December fiscal periods.

    A plain title is only an initial-version candidate. Earliest public timing,
    withdrawals, accounting scope and statement amounts need separate verification.
    """
    names = frame.report_nm.astype("string")
    annual = frame.loc[names.str.contains("사업보고서", regex=False, na=False)].copy()
    periods = annual.report_nm.astype("string").str.extract(r"사업보고서\s*\((\d{4})\.(\d{2})\)\s*$")
    annual["fiscal_year_reported"] = pd.to_numeric(periods[0], errors="coerce").astype("Int64")
    annual["fiscal_month_reported"] = pd.to_numeric(periods[1], errors="coerce").astype("Int64")
    matched = annual.fiscal_year_reported.between(STUDY_START_YEAR, STUDY_END_YEAR).fillna(False)
    valid_month = annual.fiscal_month_reported.between(1, 12).fillna(False)
    unresolved = annual.loc[annual.fiscal_year_reported.isna() | ~valid_month].copy()
    selected = annual.loc[matched & valid_month].copy()
    plain = selected.report_nm.astype("string").str.fullmatch(r"사업보고서\s*\(\d{4}\.\d{2}\)\s*", na=False)
    selected["version_kind"] = plain.map({True: "initial_candidate", False: "amendment_or_annotation"})
    selected["first_version_verified"] = False
    selected["point_in_time_values_verified"] = False
    return selected, unresolved


def main() -> int:
    """Save receipt/version coverage from complete committed disclosure requests."""
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_receipt_inventory_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    store = RequestStore(COLLECTION_STATE_PATH)
    result = {"at_kst": now.isoformat(), "cutoff": EVENT_OBSERVATION_END,
              "first_version_verified": False, "task03_complete": False}
    try:
        records = store.db.execute("SELECT request_key,status FROM requests WHERE stage='disclosures'").fetchall()
        records = [(k, s) for k, s in records if k.endswith(f"|{EVENT_OBSERVATION_END}|A")]
        frames = [store.frame("disclosures", key) for key, status in records if status == "success"]
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["report_nm"])
        selected, unresolved = annual_versions(data)
        target = INTERIM_DATA_DIR / f"annual_receipt_versions_{stamp}.csv"
        selected.to_csv(target, index=False, encoding="utf-8-sig")
        review = INTERIM_DATA_DIR / f"annual_receipt_title_review_{stamp}.csv"
        unresolved.to_csv(review, index=False, encoding="utf-8-sig")
        result["manifest"] = str(target.relative_to(PROJECT_ROOT))
        result["title_review"] = str(review.relative_to(PROJECT_ROOT))
        result["disclosure_request_statuses"] = {s: sum(status == s for _, status in records)
            for s in ("success", "unavailable", "failed")}
        result["annual_version_receipts"] = len(selected)
        result["unparsed_annual_titles"] = len(unresolved)
        result["duplicate_receipt_ids"] = int(selected.rcept_no.duplicated().sum()) if not selected.empty else 0
        result["non_december_receipts"] = int(selected.fiscal_month_reported.ne(12).sum())
        result["coverage_by_fiscal_year"] = [{"year": int(year), "receipts": len(group),
            "companies": int(group.corp_code.nunique()),
            "initial_candidates": int(group.version_kind.eq("initial_candidate").sum()),
            "amendments_or_annotations": int(group.version_kind.ne("initial_candidate").sum())}
            for year, group in selected.groupby("fiscal_year_reported")]
        log.line(f"disclosure_requests={len(records)}; annual_version_receipts={len(selected)}; unparsed_titles={len(unresolved)}; non_december_receipts={result['non_december_receipts']}; first_version_verified=False")
        for item in result["coverage_by_fiscal_year"]:
            log.line(json.dumps(item, ensure_ascii=False))
        return 0
    except Exception as exc:
        result["failure_type"] = type(exc).__name__
        log.line(f"inventory_stopped_type={type(exc).__name__}")
        return 1
    finally:
        store.close()
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.line(f"report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
