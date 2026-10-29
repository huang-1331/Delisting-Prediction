"""Collect the approved universe in resumable, rate-limited priority batches."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import json
import time
from urllib.request import urlopen

import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import RequestStore, file_sha256
from .config import (PROJECT_ROOT, INTERIM_DATA_DIR, RAW_DATA_DIR,
                     COLLECTION_STATE_PATH, EVENT_OBSERVATION_END,
                     get_dart_api_key, STUDY_START_YEAR, STUDY_END_YEAR)
from .data_collection import DartApiClient
from .reliable_collection import run_batch, CollectionIncomplete


class PacedOpener:
    """Count actual HTTP requests and enforce a local budget without logging URLs."""

    def __init__(self, budget: int, interval: float = 0.25) -> None:
        """Store a conservative per-execution budget, separate from source limits."""
        self.budget = budget
        self.interval = interval
        self.calls = 0
        self.last_start = 0.0
        self.budget_exhausted = False

    def open(self, request, timeout=30):
        """Wait between physical calls; never include request URLs in diagnostics."""
        if self.calls >= self.budget:
            self.budget_exhausted = True
            raise RuntimeError("Local HTTP budget reached; resume later")
        time.sleep(max(0.0, self.interval - (time.monotonic() - self.last_start)))
        self.last_start = time.monotonic()
        self.calls += 1
        return urlopen(request, timeout=timeout)


def save_progress(report, stamp: str, state: str, action: str) -> None:
    """Atomically record an actual collection checkpoint without replacing history."""
    path = PROJECT_ROOT / "work_progress.json"
    progress = json.loads(path.read_text(encoding="utf-8-sig"))
    task = next(item for item in progress["tasks"] if item["id"] == "03")
    task["status"] = state
    task["next_action"] = action
    rel = str(report.relative_to(PROJECT_ROOT))
    if not any(item.get("path") == rel for item in task["artifacts"]):
        task["artifacts"].append({"path": rel, "evidence": "본수집 실제 실행의 배치 집계·물리 요청 수·잔여 상태. 원시 데이터 없음."})
        task["artifacts"].append({"path": str(report.with_suffix(".txt").relative_to(PROJECT_ROOT)),
                                  "evidence": "실행 stdout/stderr 동시 저장 원문"})
    progress["updated_at"] = stamp
    progress["next_task"] = "Task03 근거 수집·최초본·업종·라벨 입력 검증 계속. Task04 진입 보류."
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    """Run historical-first collection, retaining every unknown and all checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-budget", type=int, default=18000)
    parser.add_argument("--phase", choices=["all", "historical_disclosures", "historical_statements",
        "remaining_disclosures", "remaining_statements"], default="all")
    args = parser.parse_args()
    if args.http_budget < 1:
        parser.error("HTTP budget must be positive")
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_collection_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    linked_path = INTERIM_DATA_DIR / "collection_v2_linked_2026-10-28_8dc723d3e083.csv"
    result = {"at_kst": now.isoformat(), "event_cutoff": EVENT_OBSERVATION_END,
        "candidate_path": str(linked_path.relative_to(PROJECT_ROOT)),
        "state_path": str(COLLECTION_STATE_PATH.relative_to(PROJECT_ROOT)),
        "phase": args.phase, "batches": [], "http_calls": 0,
        "raw_collection_complete": False, "task03_complete": False,
        "first_version_verified": False, "industry_verified": False,
        "http_budget": args.http_budget}
    opener = PacedOpener(args.http_budget)
    started = time.monotonic()
    originals = [*RAW_DATA_DIR.glob("dart_*.csv"), *RAW_DATA_DIR.glob("kind_*.csv"),
        *RAW_DATA_DIR.glob("krx_*.csv"), *INTERIM_DATA_DIR.glob("dart_*checkpoint*.json")]
    before = {path: file_sha256(path) for path in originals}
    try:
        progress = json.loads((PROJECT_ROOT / "work_progress.json").read_text(encoding="utf-8-sig"))
        if next(item for item in progress["tasks"] if item["id"] == "02")["status"] != "completed":
            raise ValueError("Task02 must be completed")
        reviews = sorted((PROJECT_ROOT / "reports/monthly").glob("task03_event_review_*.json"))
        verified = [json.loads(p.read_text(encoding="utf-8")) for p in reviews]
        if not any(r.get("cutoff") == EVENT_OBSERVATION_END and r.get("snapshot_agreement_verified")
                   and r.get("source_preserved") for r in verified):
            raise ValueError("Frozen event coverage is not verified")
        linked = pd.read_csv(linked_path, dtype="string")
        ready = linked.loc[linked.collection_status.eq("ready")]
        historical = set(ready.loc[ready.delisting_date.notna(), "stock_code"])
        remaining = set(ready.stock_code) - historical
        result["identity_ready_companies"] = len(historical) + len(remaining)
        result["historical_companies"] = len(historical)
        result["remaining_companies"] = len(remaining)
        log.line(f"scope=raw_evidence_only; event_cutoff={EVENT_OBSERVATION_END}; companies={result['identity_ready_companies']}; historical={len(historical)}; industry_unknown_preserved=True")
        log.line("API list versions include originals and corrections; statement snapshots remain point_in_time_verified=False")
        client = DartApiClient(get_dart_api_key(required=True), opener=opener)
        phases = [("historical_disclosures", historical, "disclosures"),
                  ("historical_statements", historical, "statements"),
                  ("remaining_disclosures", remaining, "disclosures"),
                  ("remaining_statements", remaining, "statements")]
        save_progress(report, now.isoformat(), "running", "Task03 본수집 실행 중. 배치 단위 상태보존, Task04 미실행.")
        for phase, codes, stage in phases:
            if args.phase not in {"all", phase}:
                continue
            ordered = sorted(codes)
            for offset in range(0, len(ordered), 25):
                group = set(ordered[offset:offset + 25])
                log.line(f"phase={phase}; companies={offset + 1}-{min(offset + 25, len(ordered))}/{len(ordered)}; physical_http_calls={opener.calls}", timestamp=True)
                batch = run_batch(linked_path, date.fromisoformat(EVENT_OBSERVATION_END), stage,
                    log, client=client, stock_codes=group, state_path=COLLECTION_STATE_PATH)
                entry = asdict(batch)
                entry["output"] = str(batch.output.relative_to(PROJECT_ROOT))
                entry["phase"] = phase
                entry["company_offset"] = offset
                result["batches"].append(entry)
                result["http_calls"] = opener.calls
                result["elapsed_seconds"] = round(time.monotonic() - started, 3)
                report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                save_progress(report, datetime.now(timezone(timedelta(hours=9))).isoformat(),
                    "running", f"Task03 {phase}: {min(offset + 25,len(ordered))}/{len(ordered)}기업. 동일 명령으로 성공 상태 재사용하여 재개. Task04 미실행.")
                if batch.failed > batch.recovered:
                    raise CollectionIncomplete("Retryable failures remain; stop for review")
        result["selected_phases_exhausted"] = True
        result["raw_collection_complete"] = args.phase == "all"
        return 0
    except KeyboardInterrupt:
        result["stop_reason"] = "KeyboardInterrupt"
        return 130
    except Exception as exc:
        result["stop_reason"] = "local_http_budget" if opener.budget_exhausted else type(exc).__name__
        log.line(f"collection_stopped_type={result['stop_reason']}; raw_checkpoints_preserved=True")
        return 1
    finally:
        result["http_calls"] = opener.calls
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["original_files_preserved"] = all(file_sha256(p) == h for p, h in before.items())
        snapshot = RequestStore(COLLECTION_STATE_PATH)
        result["checkpoint_counts_all_ranges"] = [dict(zip(["stage", "status", "count"], row))
            for row in snapshot.db.execute("SELECT stage,status,count(*) FROM requests GROUP BY stage,status")]
        result["statement_year_coverage"] = [dict(zip(["year", "status", "count"], row))
            for row in snapshot.db.execute("SELECT substr(request_key,17,4),status,count(*) FROM requests WHERE stage='statements' GROUP BY substr(request_key,17,4),status ORDER BY 1,2")]
        snapshot.close()
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        state = "blocked" if "stop_reason" in result else "running"
        save_progress(report, datetime.now(timezone(timedelta(hours=9))).isoformat(), state,
            "Task03 근거 수집 미완료. 동일 실행명령으로 성공 재사용·실패 재시도. 원문 최초본·T업종·라벨 입력 검증 후에만 완료 판정. Task04 미실행.")
        log.line(f"physical_http_calls={opener.calls}; elapsed_seconds={result['elapsed_seconds']}; original_files_preserved={result['original_files_preserved']}; task03_complete=False; report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
