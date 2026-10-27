"""Run the official data collection pipeline with resumable progress logs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import sys
import time
from typing import TextIO

import pandas as pd

from .config import PROJECT_ROOT, STUDY_END_YEAR, STUDY_START_YEAR
from .data_collection import (
    DartApiError,
    KrxDataClient,
    build_kosdaq_manufacturing_history,
    build_collection_candidates,
    create_dart_client,
    fetch_annual_statement_with_fallback,
    fetch_kind_current_kosdaq_listings,
    link_listing_history_to_dart_codes,
)


RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
REPORT_DIR = PROJECT_ROOT / "reports" / "monthly"


class TeeStream:
    """Mirror a terminal stream to the collection log without altering text."""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        """Store the original terminal stream and destination log stream."""

        self._terminal = terminal
        self._log_file = log_file

    def write(self, text: str) -> int:
        """Write exactly the same text to the terminal and log file."""

        self._terminal.write(text)
        self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        """Flush both output destinations."""

        self._terminal.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        """Preserve the original terminal capability flag."""

        return self._terminal.isatty()


class ProgressLog:
    """Write concise collection progress to the terminal and a text file."""

    def __init__(self, path: Path) -> None:
        """Create a fresh UTF-8 log and tee stdout and stderr into it."""

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # Refuse to overwrite an earlier direct execution log.
        self._stream = path.open("x", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = TeeStream(self._stdout, self._stream)
        sys.stderr = TeeStream(self._stderr, self._stream)

    def close(self) -> None:
        """Restore terminal streams and close the report stream."""

        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._stream.close()

    def line(self, message: str = "", *, timestamp: bool = False) -> None:
        """Print one line and mirror it to the report file."""

        rendered = message
        if timestamp and message:
            rendered = f"[{datetime.now():%H:%M:%S}] {message}"
        print(rendered, flush=True)

    def section(self, step: str, title: str) -> None:
        """Print a stable section heading suitable for screenshots."""

        self.line()
        self.line("-" * 76)
        self.line(f"[{step}] {title}")
        self.line("-" * 76)


@dataclass
class TaskCounts:
    """Running status counts for a resumable API collection stage."""

    completed: int = 0
    success: int = 0
    no_data: int = 0
    errors: int = 0
    skipped: int = 0


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a dataframe atomically as a UTF-8 CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def prepare_candidate_files(current: pd.DataFrame, delisted: pd.DataFrame,
                            corp_codes: pd.DataFrame, as_of: date,
                            log: ProgressLog) -> dict[str, Path]:
    """Save collection candidates and unresolved identities, never final industry labels."""
    from uuid import uuid4
    stamp = f"{as_of.isoformat()}_{uuid4().hex[:12]}"
    candidates = build_collection_candidates(current, delisted)
    linked = link_listing_history_to_dart_codes(candidates, corp_codes)
    paths = {name: INTERIM_DIR / f"collection_v2_{name}_{stamp}.csv"
             for name in ("candidates", "linked", "unresolved", "review")}
    _write_csv(candidates, paths["candidates"])
    _write_csv(linked.linked, paths["linked"])
    _write_csv(linked.unresolved, paths["unresolved"])
    review = pd.concat([candidates.loc[candidates.collection_status.eq("identity_review")],
                        linked.unresolved], ignore_index=True).drop_duplicates()
    _write_csv(review, paths["review"])
    ready = linked.linked.loc[linked.linked.collection_status.eq("ready")]
    log.line(f"collection_candidates={len(candidates)}; identity_ready_linked={len(ready)}; historical_ready_linked={int(ready.delisting_date.notna().sum())}; industry_confirmed=0; review={len(review)}")
    return paths


def collect_universe(as_of: date, log: ProgressLog,
                     source_date: date | None = None) -> dict[str, Path]:
    """Collect and link official KIND, KRX, and OpenDART universe records."""

    if source_date is not None:
        source_stamp = source_date.isoformat()
        current = pd.read_csv(RAW_DIR / f"kind_kosdaq_listings_{source_stamp}.csv", dtype="string")
        delisted = pd.read_csv(RAW_DIR / f"krx_kosdaq_delisted_2015_{source_stamp}.csv", dtype="string")
        corp_codes = pd.read_csv(RAW_DIR / f"dart_corp_codes_{source_stamp}.csv", dtype="string")
        log.line(f"source_snapshot={source_stamp}; collection_as_of={as_of}; source_freshness_not_extended=True")
        return prepare_candidate_files(current, delisted, corp_codes, as_of, log)
    stamp = as_of.isoformat()
    paths = {
        "kind": RAW_DIR / f"kind_kosdaq_listings_{stamp}.csv",
        "delisted": RAW_DIR / f"krx_kosdaq_delisted_2015_{stamp}.csv",
        "corp_codes": RAW_DIR / f"dart_corp_codes_{stamp}.csv",
        "history": INTERIM_DIR / f"kosdaq_manufacturing_history_{stamp}.csv",
        "linked": INTERIM_DIR / f"kosdaq_manufacturing_dart_linked_{stamp}.csv",
        "unresolved": INTERIM_DIR / f"kosdaq_manufacturing_dart_unresolved_{stamp}.csv",
    }
    if any(paths[name].exists() for name in ("kind", "delisted", "corp_codes")):
        raise ValueError("Source snapshot already exists; use --source-date to reuse it")

    log.section("1/3", "공식 기업 목록 및 상장폐지 이력 수집")
    log.line("KIND 현재 KOSDAQ 상장사 목록 요청", timestamp=True)
    current = fetch_kind_current_kosdaq_listings()
    _write_csv(current, paths["kind"])
    log.line(
        f"완료: {len(current):,}행, 고유 종목코드 {current['stock_code'].nunique():,}개"
    )

    log.line("KRX 상장폐지 이력 연도별 요청", timestamp=True)
    krx_client = KrxDataClient()
    delisted_frames: list[pd.DataFrame] = []
    for year in range(STUDY_START_YEAR, as_of.year + 1):
        period_start = date(year, 1, 1)
        period_end = min(date(year, 12, 31), as_of)
        if period_start > period_end:
            continue
        annual = krx_client.fetch_delisted_issues(
            period_start, period_end, market="KSQ"
        )
        delisted_frames.append(annual)
        log.line(f"  {year}: {len(annual):,}행")
    delisted = pd.concat(delisted_frames, ignore_index=True)
    delisted = delisted.drop_duplicates(
        subset=["stock_code", "listing_date", "delisting_date"], keep="last"
    ).reset_index(drop=True)
    _write_csv(delisted, paths["delisted"])
    log.line(
        f"완료: {len(delisted):,}행, 고유 종목코드 {delisted['stock_code'].nunique():,}개"
    )

    log.line("OpenDART 기업 고유번호 전체 목록 요청", timestamp=True)
    corp_codes = create_dart_client().fetch_corp_codes()
    _write_csv(corp_codes, paths["corp_codes"])
    log.line(f"완료: {len(corp_codes):,}행")

    return prepare_candidate_files(current, delisted, corp_codes, as_of, log)


def _company_targets(linked: pd.DataFrame) -> pd.DataFrame:
    """Return one DART target per corporation and stock code."""

    targets = linked[["corp_code", "stock_code"]].dropna().drop_duplicates()
    return targets.sort_values(["stock_code", "corp_code"]).reset_index(drop=True)


def _progress_message(
    counts: TaskCounts,
    total: int,
    started: float,
) -> str:
    """Format a compact, report-ready progress line."""

    elapsed = max(time.monotonic() - started, 0.001)
    processed = counts.completed + counts.skipped
    percent = processed / total * 100 if total else 100.0
    rate = counts.completed / elapsed * 60
    return (
        f"진행 {processed:,}/{total:,} ({percent:5.1f}%) | "
        f"성공 {counts.success:,} | 자료없음 {counts.no_data:,} | "
        f"오류 {counts.errors:,} | 재개건너뜀 {counts.skipped:,} | "
        f"분당 {rate:.1f}건"
    )


def collect_statements(linked_path: Path, as_of: date, log: ProgressLog, **options) -> Path:
    """Collect unverified raw annual snapshots using transactional restart state."""
    from .reliable_collection import run_batch, CollectionIncomplete
    result = run_batch(linked_path, as_of, "statements", log, **options)
    if result.failed:
        raise CollectionIncomplete("Statement requests failed; retry on next run")
    return result.output


def collect_disclosures(linked_path: Path, as_of: date, log: ProgressLog, **options) -> Path:
    """Collect receipt lists with range-aware request identity and safe retries."""
    from .reliable_collection import run_batch, CollectionIncomplete
    result = run_batch(linked_path, as_of, "disclosures", log, **options)
    if result.failed:
        raise CollectionIncomplete("Disclosure requests failed; retry on next run")
    return result.output

def _parse_args() -> argparse.Namespace:
    """Parse collection stage, date, and report-log options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("universe", "statements", "disclosures", "all"),
        default="all",
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--source-date", type=date.fromisoformat,
                        help="Rebuild candidates from an existing dated source snapshot")
    parser.add_argument("--linked-path", type=Path,
                        help="Explicit v2 linked candidates when not rebuilding sources")
    parser.add_argument("--state", type=Path, help="Stable SQLite state, including after filename changes")
    parser.add_argument("--max-requests", type=int, help="Bound new requests for a smoke test")
    parser.add_argument("--years", type=int, nargs="+")
    parser.add_argument("--stock-codes", nargs="+")
    parser.add_argument("--retry-unavailable", action="store_true")
    default_log = REPORT_DIR / (
        f"collection_runner_{datetime.now():%Y%m%d_%H%M%S_%f}.txt"
    )
    parser.add_argument("--log", type=Path, default=default_log)
    return parser.parse_args()


def main() -> int:
    """Execute selected stages and preserve a report-ready terminal log."""

    args = _parse_args()
    log = ProgressLog(args.log)
    started = time.monotonic()
    stamp = args.as_of.isoformat()
    linked_path = args.linked_path
    try:
        log.line("=" * 76)
        log.line("KOSDAQ 제조업 상장폐지 예측 프로젝트 - 공식 데이터 실수집")
        log.line("=" * 76)
        log.line(f"수집 기준일: {args.as_of}")
        log.line(f"연구 사업연도: {STUDY_START_YEAR}~{STUDY_END_YEAR}")
        log.line("인증정보는 새 요청이 필요할 때 로컬 파일에서 읽습니다.")

        if args.stage in {"universe", "all"} or args.source_date is not None:
            paths = collect_universe(args.as_of, log, args.source_date)
            linked_path = paths["linked"]
        options = {"max_requests": args.max_requests, "years": args.years,
                   "stock_codes": set(args.stock_codes) if args.stock_codes else None,
                   "retry_unavailable": args.retry_unavailable}
        if args.state is not None:
            options["state_path"] = args.state
        if args.stage in {"statements", "all"}:
            if linked_path is None or not linked_path.exists():
                raise FileNotFoundError("Provide --linked-path or --source-date")
            collect_statements(linked_path, args.as_of, log, **options)
        if args.stage in {"disclosures", "all"}:
            if linked_path is None or not linked_path.exists():
                raise FileNotFoundError("Provide --linked-path or --source-date")
            collect_disclosures(linked_path, args.as_of, log, **options)

        elapsed_minutes = (time.monotonic() - started) / 60
        log.section("완료", "수집 실행 결과")
        log.line(f"상태: PASS")
        log.line(f"총 실행시간: {elapsed_minutes:.1f}분")
        log.line(f"터미널 기록: {args.log}")
        return 0
    except KeyboardInterrupt:
        elapsed_minutes = (time.monotonic() - started) / 60
        log.section("중단", "수집 실행 결과")
        log.line("상태: INTERRUPTED")
        log.line(f"실행시간: {elapsed_minutes:.1f}분")
        log.line("완료된 요청은 체크포인트에 보존되어 재실행 시 건너뜁니다.")
        return 130
    except Exception as exc:
        elapsed_minutes = (time.monotonic() - started) / 60
        log.section("중단", "수집 실행 결과")
        log.line(f"상태: INCOMPLETE")
        # Exception text can contain a URL with an API key; never print it.
        log.line(f"중단 사유 유형: {type(exc).__name__}")
        log.line(f"실행시간: {elapsed_minutes:.1f}분")
        log.line("완료된 요청은 체크포인트에 보존되어 재실행 시 건너뜁니다.")
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
