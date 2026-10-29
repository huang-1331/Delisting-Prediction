"""Verify the frozen event horizon against official, resumably collected records."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import time

import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import RequestStore, file_sha256
from .config import (PROJECT_ROOT, RAW_DATA_DIR, INTERIM_DATA_DIR,
                     EVENT_OBSERVATION_END, STUDY_START_YEAR)
from .data_collection import KrxDataClient
from .labeling import classify_delisting_reasons, list_unresolved_delisting_reasons


def canonical_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifiers and dates for comparison, retaining every source field."""
    columns = ["stock_code", "company_name", "market", "industry", "listing_date",
               "delisting_date", "delisting_reason", "security_type", "stock_type"]
    result = frame[columns].copy().astype("string")
    for column in ("listing_date", "delisting_date"):
        result[column] = pd.to_datetime(result[column], errors="raise").dt.strftime("%Y-%m-%d")
    # CSV readers represent empty source cells as NA; JSON retains "". Compare
    # the same absent text consistently, without assigning research values.
    return result.replace("", pd.NA).fillna("<missing>")


def main() -> int:
    """Collect annual event responses, compare the frozen snapshot and log evidence."""
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_event_review_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    cutoff = date.fromisoformat(EVENT_OBSERVATION_END)
    source = RAW_DATA_DIR / f"krx_kosdaq_delisted_2015_{cutoff}.csv"
    state = RequestStore(INTERIM_DATA_DIR / "event_evidence_v1.sqlite3")
    result = {"at_kst": now.isoformat(), "cutoff": str(cutoff),
              "source_snapshot": str(source.relative_to(PROJECT_ROOT)),
              "source_sha256_before": file_sha256(source),
              "new_success": 0, "reused": 0, "unavailable": 0, "failed": 0,
              "year_coverage": [], "snapshot_agreement_verified": False,
              "label_inputs_complete": False}
    started = time.monotonic()
    client = None
    keys = []
    frames = []
    try:
        for year in range(STUDY_START_YEAR, cutoff.year + 1):
            start, end = date(year, 1, 1), min(date(year, 12, 31), cutoff)
            key = f"KSQ|{start}|{end}|MDCSTAT23801"

            def fetch() -> pd.DataFrame:
                """Fetch one bounded official interval; unknown reasons remain raw."""
                nonlocal client
                if client is None:
                    client = KrxDataClient()
                return client.fetch_delisted_issues(start, end, market="KSQ")

            status, reused, code = state.run("events", key, fetch)
            keys.append(key)
            result["reused"] += int(reused)
            if not reused:
                result[{"success": "new_success", "unavailable": "unavailable", "failed": "failed"}[status]] += 1
            log.line(f"event_year={year}; start={start}; end={end}; status={status}; reused={reused}; error_type={code or 'none'}")
            if status != "success":
                result["stop_reason"] = "official_interval_not_verified"
                return 1
            frame = state.frame("events", key)
            dates = pd.to_datetime(frame.delisting_date, errors="raise")
            if not dates.between(pd.Timestamp(start), pd.Timestamp(end)).all():
                raise ValueError("Event response outside the requested interval")
            result["year_coverage"].append({"year": year, "rows": len(frame),
                "earliest_event": dates.min().isoformat(), "latest_event": dates.max().isoformat()})
            frames.append(frame)
            time.sleep(1)
        output = RAW_DATA_DIR / f"event_verification_{cutoff}_{stamp}.csv"
        state.export("events", keys, output)
        result["official_export"] = str(output.relative_to(PROJECT_ROOT))
        official = pd.concat(frames, ignore_index=True)
        legacy = pd.read_csv(source, dtype="string")
        old, new = canonical_events(legacy), canonical_events(official)
        identity = ["stock_code", "listing_date", "delisting_date"]
        result["old_rows"], result["official_rows"] = len(old), len(new)
        result["old_duplicate_identities"] = int(old.duplicated(identity).sum())
        result["official_duplicate_identities"] = int(new.duplicated(identity).sum())
        differences = old.merge(new, how="outer", on=list(old.columns), indicator=True)
        changed = differences.loc[differences._merge.ne("both")]
        review = INTERIM_DATA_DIR / f"event_snapshot_differences_{stamp}.csv"
        changed.to_csv(review, index=False, encoding="utf-8-sig")
        result["difference_review"] = str(review.relative_to(PROJECT_ROOT))
        result["old_only_rows"] = int(differences._merge.eq("left_only").sum())
        result["official_only_rows"] = int(differences._merge.eq("right_only").sum())
        result["snapshot_agreement_verified"] = (changed.empty and not result["old_duplicate_identities"]
                                                  and not result["official_duplicate_identities"])
        common = official.loc[official.security_type.eq("주권") & official.stock_type.eq("보통주")]
        classified = classify_delisting_reasons(common.delisting_reason)
        result["common_equity_events"] = len(common)
        result["reason_rule_distress"] = int(classified.eq(True).sum())
        result["reason_rule_competing"] = int(classified.eq(False).sum())
        result["reason_unresolved_events"] = int(classified.isna().sum())
        unresolved = list_unresolved_delisting_reasons(common)
        unresolved_path = INTERIM_DATA_DIR / f"event_unresolved_reasons_{stamp}.csv"
        unresolved.to_csv(unresolved_path, index=False, encoding="utf-8-sig")
        result["unresolved_reason_review"] = str(unresolved_path.relative_to(PROJECT_ROOT))
        log.line(f"snapshot_agreement={result['snapshot_agreement_verified']}; old_rows={len(old)}; official_rows={len(new)}; old_only={result['old_only_rows']}; official_only={result['official_only_rows']}")
        log.line(f"common_equity_events={len(common)}; unresolved_reason_events={result['reason_unresolved_events']}; label_inputs_complete=False")
        return 0 if result["snapshot_agreement_verified"] else 2
    except Exception as exc:
        result["stop_reason"] = type(exc).__name__
        log.line(f"review_stopped_type={type(exc).__name__}")
        return 1
    finally:
        result["source_sha256_after"] = file_sha256(source)
        result["source_preserved"] = result["source_sha256_before"] == result["source_sha256_after"]
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        state.close()
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.line(f"source_preserved={result['source_preserved']}; elapsed_seconds={result['elapsed_seconds']}; report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
