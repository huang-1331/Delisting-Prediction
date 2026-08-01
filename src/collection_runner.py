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
        self._stream = path.open("w", encoding="utf-8")
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


def _load_completed(path: Path) -> set[str]:
    """Load completed task identifiers from a checkpoint JSON file."""

    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("completed", []))


def _save_completed(path: Path, completed: set[str]) -> None:
    """Atomically save completed task identifiers without credential data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"completed": sorted(completed)}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_csv(frame: pd.DataFrame, path: Path) -> None:
    """Append a non-empty dataframe while writing the header only once."""

    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists() or path.stat().st_size == 0,
        index=False,
        encoding="utf-8-sig",
    )


def collect_universe(as_of: date, log: ProgressLog) -> dict[str, Path]:
    """Collect and link official KIND, KRX, and OpenDART universe records."""

    stamp = as_of.isoformat()
    paths = {
        "kind": RAW_DIR / f"kind_kosdaq_listings_{stamp}.csv",
        "delisted": RAW_DIR / f"krx_kosdaq_delisted_2015_{stamp}.csv",
        "corp_codes": RAW_DIR / f"dart_corp_codes_{stamp}.csv",
        "history": INTERIM_DIR / f"kosdaq_manufacturing_history_{stamp}.csv",
        "linked": INTERIM_DIR / f"kosdaq_manufacturing_dart_linked_{stamp}.csv",
        "unresolved": INTERIM_DIR / f"kosdaq_manufacturing_dart_unresolved_{stamp}.csv",
    }

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

    history = build_kosdaq_manufacturing_history(current, delisted)
    linked = link_listing_history_to_dart_codes(history, corp_codes)
    _write_csv(history, paths["history"])
    _write_csv(linked.linked, paths["linked"])
    _write_csv(linked.unresolved, paths["unresolved"])

    total = len(history)
    link_rate = (len(linked.linked) / total * 100) if total else 0.0
    log.line(f"KOSDAQ 제조업 연구대상 이력: {total:,}행")
    log.line(f"DART 연결 성공: {len(linked.linked):,}행 ({link_rate:.1f}%)")
    log.line(f"수동 검토 대상: {len(linked.unresolved):,}행")
    return paths


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


def collect_statements(
    linked_path: Path,
    as_of: date,
    log: ProgressLog,
) -> Path:
    """Collect annual statements with a company-year resume checkpoint."""

    stamp = as_of.isoformat()
    output = RAW_DIR / f"dart_annual_statements_{STUDY_START_YEAR}_{STUDY_END_YEAR}_{stamp}.csv"
    issues_path = RAW_DIR / f"dart_annual_statement_issues_{stamp}.csv"
    checkpoint = INTERIM_DIR / f"dart_annual_statement_checkpoint_{stamp}.json"
    linked = pd.read_csv(
        linked_path,
        dtype={"corp_code": "string", "stock_code": "string"},
    )
    companies = _company_targets(linked)
    years = range(STUDY_START_YEAR, STUDY_END_YEAR + 1)
    tasks = [
        (str(row.corp_code), str(row.stock_code), year)
        for row in companies.itertuples(index=False)
        for year in years
    ]
    completed_ids = _load_completed(checkpoint)
    counts = TaskCounts()
    started = time.monotonic()
    client = create_dart_client()

    log.section("2/3", "OpenDART 2015~2024 사업보고서 재무계정 수집")
    log.line(
        f"대상 기업 {len(companies):,}개, 회사-연도 요청 {len(tasks):,}건 "
        "(연결 CFS 우선, 별도 OFS 대체)"
    )
    for corp_code, stock_code, year in tasks:
        task_id = f"{corp_code}|{stock_code}|{year}"
        if task_id in completed_ids:
            counts.skipped += 1
            continue
        try:
            frame = fetch_annual_statement_with_fallback(client, corp_code, year)
        except DartApiError as exc:
            if exc.status == "020":
                log.line("OpenDART 일일 요청 한도 도달: 체크포인트 저장 후 중단")
                _save_completed(checkpoint, completed_ids)
                raise
            issue = pd.DataFrame(
                [{
                    "corp_code": corp_code,
                    "stock_code": stock_code,
                    "business_year": year,
                    "issue": f"OpenDART {exc.status}: {exc.message}",
                }]
            )
            _append_csv(issue, issues_path)
            counts.errors += 1
        except Exception as exc:
            issue = pd.DataFrame(
                [{
                    "corp_code": corp_code,
                    "stock_code": stock_code,
                    "business_year": year,
                    "issue": f"{type(exc).__name__}: {exc}",
                }]
            )
            _append_csv(issue, issues_path)
            counts.errors += 1
        else:
            if frame.empty:
                issue = pd.DataFrame(
                    [{
                        "corp_code": corp_code,
                        "stock_code": stock_code,
                        "business_year": year,
                        "issue": "No CFS or OFS annual statement returned",
                    }]
                )
                _append_csv(issue, issues_path)
                counts.no_data += 1
            else:
                frame["stock_code"] = stock_code
                _append_csv(frame, output)
                counts.success += 1

        completed_ids.add(task_id)
        counts.completed += 1
        if counts.completed % 25 == 0:
            _save_completed(checkpoint, completed_ids)
        if counts.completed % 100 == 0:
            log.line(_progress_message(counts, len(tasks), started), timestamp=True)

    _save_completed(checkpoint, completed_ids)
    log.line(_progress_message(counts, len(tasks), started), timestamp=True)
    log.line(f"재무계정 저장 파일: {output.name}")
    log.line(f"누락·오류 기록 파일: {issues_path.name}")
    return output


def collect_disclosures(
    linked_path: Path,
    as_of: date,
    log: ProgressLog,
) -> Path:
    """Collect disclosure receipt dates with a corporation resume checkpoint."""

    stamp = as_of.isoformat()
    output = RAW_DIR / f"dart_disclosures_{STUDY_START_YEAR}_{stamp}.csv"
    issues_path = RAW_DIR / f"dart_disclosure_issues_{stamp}.csv"
    checkpoint = INTERIM_DIR / f"dart_disclosure_checkpoint_{stamp}.json"
    linked = pd.read_csv(
        linked_path,
        dtype={"corp_code": "string", "stock_code": "string"},
    )
    companies = _company_targets(linked)
    completed_ids = _load_completed(checkpoint)
    counts = TaskCounts()
    started = time.monotonic()
    client = create_dart_client()

    log.section("3/3", "OpenDART 공시 접수일 수집")
    log.line(
        f"대상 기업 {len(companies):,}개, 조회기간 {STUDY_START_YEAR}-01-01~{as_of}"
    )
    for row in companies.itertuples(index=False):
        corp_code = str(row.corp_code)
        stock_code = str(row.stock_code)
        task_id = f"{corp_code}|{stock_code}"
        if task_id in completed_ids:
            counts.skipped += 1
            continue
        try:
            frame = client.fetch_disclosures(
                corp_code,
                date(STUDY_START_YEAR, 1, 1),
                as_of,
            )
        except DartApiError as exc:
            if exc.status == "020":
                log.line("OpenDART 일일 요청 한도 도달: 체크포인트 저장 후 중단")
                _save_completed(checkpoint, completed_ids)
                raise
            frame = pd.DataFrame()
            issue = pd.DataFrame(
                [{
                    "corp_code": corp_code,
                    "stock_code": stock_code,
                    "issue": f"OpenDART {exc.status}: {exc.message}",
                }]
            )
            _append_csv(issue, issues_path)
            if exc.status == "013":
                counts.no_data += 1
            else:
                counts.errors += 1
        except Exception as exc:
            frame = pd.DataFrame()
            issue = pd.DataFrame(
                [{
                    "corp_code": corp_code,
                    "stock_code": stock_code,
                    "issue": f"{type(exc).__name__}: {exc}",
                }]
            )
            _append_csv(issue, issues_path)
            counts.errors += 1
        else:
            if frame.empty:
                counts.no_data += 1
            else:
                frame["stock_code"] = stock_code
                _append_csv(frame, output)
                counts.success += 1

        completed_ids.add(task_id)
        counts.completed += 1
        if counts.completed % 25 == 0:
            _save_completed(checkpoint, completed_ids)
        if counts.completed % 100 == 0:
            log.line(_progress_message(counts, len(companies), started), timestamp=True)

    _save_completed(checkpoint, completed_ids)
    log.line(_progress_message(counts, len(companies), started), timestamp=True)
    log.line(f"공시 목록 저장 파일: {output.name}")
    return output


def _parse_args() -> argparse.Namespace:
    """Parse collection stage, date, and report-log options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("universe", "statements", "disclosures", "all"),
        default="all",
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    default_log = REPORT_DIR / (
        f"2026-08_collection_runner_{datetime.now():%Y%m%d_%H%M%S}.txt"
    )
    parser.add_argument("--log", type=Path, default=default_log)
    return parser.parse_args()


def main() -> int:
    """Execute selected stages and preserve a report-ready terminal log."""

    args = _parse_args()
    log = ProgressLog(args.log)
    started = time.monotonic()
    stamp = args.as_of.isoformat()
    linked_path = INTERIM_DIR / f"kosdaq_manufacturing_dart_linked_{stamp}.csv"
    try:
        log.line("=" * 76)
        log.line("KOSDAQ 제조업 상장폐지 예측 프로젝트 - 공식 데이터 실수집")
        log.line("=" * 76)
        log.line(f"수집 기준일: {args.as_of}")
        log.line(f"연구 사업연도: {STUDY_START_YEAR}~{STUDY_END_YEAR}")
        log.line("인증 확인: 로컬 비밀정보 로드 완료 (인증값은 표시하지 않음)")

        if args.stage in {"universe", "all"}:
            paths = collect_universe(args.as_of, log)
            linked_path = paths["linked"]
        if args.stage in {"statements", "all"}:
            if not linked_path.exists():
                raise FileNotFoundError(f"Linked universe not found: {linked_path}")
            collect_statements(linked_path, args.as_of, log)
        if args.stage in {"disclosures", "all"}:
            if not linked_path.exists():
                raise FileNotFoundError(f"Linked universe not found: {linked_path}")
            collect_disclosures(linked_path, args.as_of, log)

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
        log.line(f"중단 사유: {type(exc).__name__}: {exc}")
        log.line(f"실행시간: {elapsed_minutes:.1f}분")
        log.line("완료된 요청은 체크포인트에 보존되어 재실행 시 건너뜁니다.")
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
