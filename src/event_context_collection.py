"""Collect missing corporate-action evidence without changing model inputs."""

from datetime import date, datetime, timedelta, timezone
import json
import time

import pandas as pd

from .config import PROJECT_ROOT, INTERIM_DATA_DIR, EVENT_OBSERVATION_END, get_dart_api_key
from .data_collection import DartApiClient
from .collection_state import RequestStore, file_sha256
from .collection_runner import ProgressLog
from .task03_collect import PacedOpener


def main() -> int:
    """Collect all types for unresolved events, in a separate resumable store.

    The two-year lookback is a source-search window, never an eligibility or
    event-definition rule. No match means further investigation, not no event.
    """
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_event_context_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    store = RequestStore(INTERIM_DATA_DIR / "event_context_v1.sqlite3")
    opener = PacedOpener(2000)
    started = time.monotonic()
    result = {"at_kst": now.isoformat(), "success": 0, "unavailable": 0,
        "failed": 0, "reused": 0, "label_context_only": True,
        "model_inputs_changed": False, "task03_complete": False, "task04_executed": False}
    try:
        source = sorted(INTERIM_DATA_DIR.glob("event_reason_review_*.csv"))[-1]
        reviewed = pd.read_csv(source, dtype="string")
        events = reviewed.loc[reviewed.review_status.eq("unresolved_official_context_required")]
        linked = pd.read_csv(INTERIM_DATA_DIR / "collection_v2_linked_2026-10-28_8dc723d3e083.csv", dtype="string")
        client = DartApiClient(get_dart_api_key(required=True), opener=opener)
        frames = []
        result.update({"review_source": str(source.relative_to(PROJECT_ROOT)),
            "review_source_sha256": file_sha256(source), "events": len(events)})
        for row in events.itertuples(index=False):
            identity = linked.loc[linked.stock_code.eq(row.stock_code)]
            if len(identity) != 1 or identity.iloc[0].collection_status != "ready":
                raise ValueError("Unresolved company identity")
            corp = identity.iloc[0].corp_code
            event = pd.Timestamp(row.delisting_date)
            begin = (event - pd.DateOffset(years=2)).date()
            end = date.fromisoformat(EVENT_OBSERVATION_END)
            key = f"{corp}|{row.stock_code}|{begin.isoformat()}|{end.isoformat()}|ALL"

            def fetch():
                """Preserve raw list identity and receipts; no reason classification."""
                frame = client.fetch_disclosures(corp, begin, end, disclosure_type=None)
                if not frame.empty:
                    if not frame.corp_code.astype(str).eq(corp).all():
                        raise ValueError("Corporate-action source identity mismatch")
                    frame["stock_code"] = row.stock_code
                return frame

            status, reused, code = store.run("event_context_disclosures", key, fetch)
            result[status] += 1
            result["reused"] += int(reused)
            log.line(f"event_stock={row.stock_code}; outcome={status}; reused={reused}; error_code={code or 'none'}")
            if status == "failed":
                raise RuntimeError("Context request failed; preserve checkpoint")
            if status == "success":
                frame = store.frame("event_context_disclosures", key)
                frame["event_stock_code"] = row.stock_code
                frame["official_delisting_date"] = row.delisting_date
                frames.append(frame)
        full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        output = INTERIM_DATA_DIR / f"event_context_filings_{stamp}.csv"
        full.to_csv(output, index=False, encoding="utf-8-sig")
        # Search candidates only. Completion/withdrawal and event identity require review.
        candidates = full.loc[full.report_nm.str.contains("교환|이전|합병|폐지|종료|철회|예비심사", regex=True, na=False)]
        selected = INTERIM_DATA_DIR / f"event_context_candidates_{stamp}.csv"
        candidates.to_csv(selected, index=False, encoding="utf-8-sig")
        result.update({"filings": len(full), "candidate_filings": len(candidates),
            "filing_inventory": str(output.relative_to(PROJECT_ROOT)),
            "candidate_inventory": str(selected.relative_to(PROJECT_ROOT)),
            "database_integrity": store.db.execute("PRAGMA quick_check").fetchone()[0]})
        return 0
    except Exception as exc:
        result["stop_error_type"] = type(exc).__name__
        log.line(f"event_context_stopped={type(exc).__name__}; checkpoints_preserved=True")
        return 1
    finally:
        result["http_calls"] = opener.calls
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        store.close()
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.line(json.dumps(result, ensure_ascii=False))
        log.line(f"report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
