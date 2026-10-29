"""Audit actual collection coverage, unresolved evidence, and restart requirements."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import json

import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import RequestStore, statement_key, file_sha256
from .config import (PROJECT_ROOT, INTERIM_DATA_DIR, RAW_DATA_DIR,
    COLLECTION_STATE_PATH, MARKET_PROBE_STATE_PATH, EVENT_OBSERVATION_END,
    STUDY_START_YEAR, STUDY_END_YEAR)
from .labeling import (DISTRESS_REASON_KEYWORDS, NON_DISTRESS_REASON_KEYWORDS,
                      classify_delisting_reasons)
from .receipt_archive_collection import verify_archive


def main() -> int:
    """Count committed outcomes and verify sources; no API calls or label creation."""
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_coverage_review_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    result = {"at_kst": now.isoformat(), "event_cutoff": EVENT_OBSERVATION_END,
        "task03_complete": False, "raw_snapshot_is_first_version": False,
        "final_industry_observations_verified": 0, "final_label_observations_verified": 0,
        "external_requests_this_review": 0}
    financial = RequestStore(COLLECTION_STATE_PATH)
    documents = RequestStore(MARKET_PROBE_STATE_PATH)
    try:
        linked_path = INTERIM_DATA_DIR / "collection_v2_linked_2026-10-28_8dc723d3e083.csv"
        linked = pd.read_csv(linked_path, dtype="string")
        ready = linked.loc[linked.collection_status.eq("ready")]
        targets = ready[["corp_code", "stock_code"]].drop_duplicates()
        result["identity_ready_companies"] = len(targets)
        result["historical_companies"] = int(ready.loc[ready.delisting_date.notna(), "stock_code"].nunique())
        result["industry_status"] = "All identity-ready firms remain collection candidates; final T-industry not verified"
        industry_files = sorted(INTERIM_DATA_DIR.glob("industry_review_spac_*.csv"))
        verified_reviews = []
        for candidate in sorted((PROJECT_ROOT / "reports/monthly").glob("task03_verified_reviews_*.json")):
            item = json.loads(candidate.read_text(encoding="utf-8"))
            if not item.get("stop_error_type") and item.get("industry_artifact"):
                verified_reviews.append(item)
        if verified_reviews:
            reviewed = verified_reviews[-1]
            combined_path = PROJECT_ROOT / reviewed["industry_artifact"]
            if file_sha256(combined_path) != reviewed["industry_sha256"]:
                raise ValueError("Combined reviewed industry artifact changed")
            industry_files = [combined_path]
            reason_path = PROJECT_ROOT / reviewed["reason_artifact"]
            if file_sha256(reason_path) != reviewed["reason_sha256"]:
                raise ValueError("Reviewed event reason artifact changed")
            result["reviewed_event_reason_artifact"] = reviewed["reason_artifact"]
            result["reviewed_unknown_reason_counts"] = {
                "distress": reviewed["reviewed_distress_events_in_original_unknowns"],
                "competing": reviewed["new_reviewed_competing_events"],
                "unresolved": reviewed["remaining_unknown_events"]}
            result["reviewed_reason_facts_applied_to_production_classifier"] = False
        if industry_files:
            industry_path = industry_files[-1]
            industry = pd.read_csv(industry_path, dtype="string")
            if industry.duplicated(["corp_code", "stock_code", "T"]).any():
                raise ValueError("Duplicate reviewed industry observations")
            if not industry.review_status.isin(["reviewed_original_business_purpose", "reviewed_original_business_statement"]).all():
                raise ValueError("Unverified industry row included in reviewed facts")
            for row in industry.itertuples(index=False):
                if file_sha256(PROJECT_ROOT / row.source_path) != row.source_sha256:
                    raise ValueError("Reviewed industry source changed")
            result["reviewed_industry_artifact"] = str(industry_path.relative_to(PROJECT_ROOT))
            result["final_industry_observations_verified"] = len(industry)
            result["verified_nonmanufacturing_observations"] = int(industry.manufacturing_confirmed.eq("0").sum())
            result["verified_manufacturing_observations"] = int(industry.manufacturing_confirmed.eq("1").sum())
            result["industry_status"] = "Only explicitly reviewed company-T facts are verified; remaining industry observations pending"
        result["database_integrity"] = financial.db.execute("PRAGMA quick_check").fetchone()[0]
        states = {(stage, key): status for stage, key, status in financial.db.execute(
            "SELECT stage,request_key,status FROM requests").fetchall()}
        result["failed_request_error_codes"] = [dict(zip(["stage", "error_code", "count"], row))
            for row in financial.db.execute("SELECT stage,error_code,count(*) FROM requests WHERE status='failed' GROUP BY stage,error_code")]
        year_coverage = []
        scope_keys = []
        for year in range(STUDY_START_YEAR, STUDY_END_YEAR + 1):
            keys = [statement_key(row.corp_code, row.stock_code, year) for row in targets.itertuples(index=False)]
            scope_keys.extend(keys)
            count = Counter(states.get(("statements", key), "not_attempted") for key in keys)
            entry = {"year": year, **{s: count[s] for s in ("success", "unavailable", "failed", "not_attempted")}}
            year_coverage.append(entry)
            log.line(f"financial_year={year}; " + "; ".join(f"{k}={v}" for k, v in entry.items() if k != "year"))
        result["financial_year_coverage"] = year_coverage
        totals = Counter(states.get(("statements", key), "not_attempted") for key in scope_keys)
        result["financial_totals"] = {s: totals[s] for s in ("success", "unavailable", "failed", "not_attempted")}
        recoveries = []
        for key in scope_keys:
            if states.get(("statements", key)) == "failed":
                manifest = financial.original_recovery(key)
                if manifest:
                    recoveries.append({"request_key": key, "original_receipt": manifest["original_receipt"],
                        "T": manifest["T"], "canonical_account_mapping": manifest["canonical_account_mapping"]})
        result["api_failures_with_verified_original_recovery"] = recoveries
        result["unresolved_financial_failures"] = totals["failed"] - len(recoveries)
        result["remaining_financial_serial_spacing_only_seconds"] = totals["not_attempted"] * 0.25
        result["remaining_time_caveat"] = "Spacing-only lower bound for serial raw API requests; network, retries, disclosure pages, undiscovered original receipts and verification are additional. Not a promise or reduced scope."
        disclosure_counts = Counter()
        disclosure_frames = []
        for row in targets.itertuples(index=False):
            key = f"{row.corp_code}|{row.stock_code}|{STUDY_START_YEAR}-01-01|{EVENT_OBSERVATION_END}|A"
            status = states.get(("disclosures", key), "not_attempted")
            disclosure_counts[status] += 1
            if status == "success":
                disclosure_frames.append(financial.frame("disclosures", key))
        result["disclosure_totals"] = {s: disclosure_counts[s] for s in ("success", "unavailable", "failed", "not_attempted")}
        disclosure = pd.concat(disclosure_frames, ignore_index=True) if disclosure_frames else pd.DataFrame()
        if not disclosure.empty:
            result["disclosure_rows"] = len(disclosure)
            result["duplicate_disclosure_receipts"] = int(disclosure.rcept_no.duplicated().sum())
            result["disclosures_after_event_cutoff"] = int((pd.to_datetime(disclosure.rcept_dt) > pd.Timestamp(EVENT_OBSERVATION_END)).sum())
        result["financial_rows"] = 0
        api_receipts = set()
        for key in scope_keys:
            if states.get(("statements", key)) != "success":
                continue
            frame = financial.frame("statements", key)
            result["financial_rows"] += len(frame)
            api_receipts.update(frame.rcept_no.astype(str))
        result["api_snapshot_receipt_ids"] = len(api_receipts)
        manifests = sorted(INTERIM_DATA_DIR.glob("annual_receipt_versions_*.csv"))
        if manifests:
            manifest_path = manifests[-1]
            manifest = pd.read_csv(manifest_path, dtype="string")
            result["receipt_manifest"] = str(manifest_path.relative_to(PROJECT_ROOT))
            result["manifest_rows"] = len(manifest)
            receipt_status = {k: s for k, s in documents.db.execute(
                "SELECT request_key,status FROM requests WHERE stage='dart_receipt_original'").fetchall()}
            archive_counts = Counter()
            for receipt in manifest.rcept_no.dropna().drop_duplicates():
                status = receipt_status.get(receipt, "not_attempted")
                archive_counts[status] += 1
                if status == "success":
                    metadata = documents.frame("dart_receipt_original", receipt).iloc[0]
                    path = PROJECT_ROOT / metadata.path
                    if not path.resolve().is_relative_to(RAW_DATA_DIR.resolve()):
                        raise ValueError("Archive metadata path outside raw data")
                    verify_archive(path, metadata.sha256)
            result["archive_statuses_for_manifest"] = dict(archive_counts)
            result["archive_year_coverage"] = []
            for year, group in manifest.groupby("fiscal_year_reported"):
                count = Counter(receipt_status.get(r, "not_attempted") for r in group.rcept_no.dropna().drop_duplicates())
                result["archive_year_coverage"].append({"year": int(year), **dict(count)})
            first_candidates = manifest.loc[manifest.version_kind.eq("initial_candidate")]
            result["api_receipts_matching_initial_title_candidates"] = len(api_receipts & set(first_candidates.rcept_no))
            result["api_receipts_not_matching_initial_title_candidates"] = len(api_receipts - set(first_candidates.rcept_no))
            result["matching_receipt_implies_verified_values"] = False
        source = RAW_DATA_DIR / f"krx_kosdaq_delisted_2015_{EVENT_OBSERVATION_END}.csv"
        events = pd.read_csv(source, dtype="string")
        common = events.loc[events.security_type.eq("주권") & events.stock_type.eq("보통주")]
        classified = classify_delisting_reasons(common.delisting_reason)
        reasons = common.delisting_reason.fillna("").str.replace(r"\s+", "", regex=True)
        mixed = reasons.apply(lambda value: any(k.replace(" ", "") in value for k in DISTRESS_REASON_KEYWORDS)
            and any(k.replace(" ", "") in value for k in NON_DISTRESS_REASON_KEYWORDS))
        result["event_common_equity_rows"] = len(common)
        result["event_unknown_reason_rows"] = int(classified.isna().sum())
        result["event_keyword_overlap_rows"] = int(mixed.sum())
        result["keyword_overlap_is_verified_competing_event"] = False
        mixed_path = INTERIM_DATA_DIR / f"event_keyword_overlap_review_{stamp}.csv"
        common.loc[mixed].to_csv(mixed_path, index=False, encoding="utf-8-sig")
        result["event_keyword_overlap_review"] = str(mixed_path.relative_to(PROJECT_ROOT))
        result["event_source_sha256"] = file_sha256(source)
        result["required_reviews"] = ["T-industry and historical manufacturing classification evidence",
            "First public statement values, accounting scope, currency and Altman account mapping",
            "Unknown/mixed delisting reasons and receipt/listing eligibility at T",
            "Uncollected candidates, unavailable essentials and collection failures"]
        result["resume_commands"] = ["python -m src.task03_collect --phase all",
            "python -m src.annual_receipt_inventory",
            "python -m src.receipt_archive_collection --manifest <verified latest manifest path>",
            "python -m src.task03_coverage_review"]
        log.line(f"financial_totals={result['financial_totals']}; disclosure_totals={result['disclosure_totals']}; archives={result.get('archive_statuses_for_manifest',{})}")
        log.line(f"event_unknown_reasons={result['event_unknown_reason_rows']}; event_keyword_overlap_rows={result['event_keyword_overlap_rows']}; task03_complete=False; task04_executed=False")
        return 0
    except Exception as exc:
        result["review_error_type"] = type(exc).__name__
        log.line(f"review_error_type={type(exc).__name__}")
        return 1
    finally:
        financial.close()
        documents.close()
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.line(f"report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
