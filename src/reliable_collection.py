"""Small, resumable raw-collection batches; final study eligibility is separate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import time
from typing import Any, Iterable
from uuid import uuid4

import pandas as pd

from .collection_state import RequestStore, statement_key
from .config import (COLLECTION_STATE_PATH, COLLECTION_MAX_ATTEMPTS,
                     COLLECTION_RETRY_SECONDS, RAW_DATA_DIR, INTERIM_DATA_DIR,
                     STUDY_START_YEAR, STUDY_END_YEAR)
from .data_collection import create_dart_client, fetch_annual_statement_with_fallback


class CollectionIncomplete(RuntimeError):
    """Signal incomplete collection using a credential-free status code."""


@dataclass
class BatchResult:
    """Counts of final outcomes for one explicitly selected request batch."""
    output: Path
    success: int = 0
    unavailable: int = 0
    failed: int = 0
    skipped: int = 0
    requests: int = 0
    rows: int = 0
    recovered: int = 0


def run_batch(linked_path: Path, as_of: date, stage: str, log: Any, *,
              client: Any = None, years: Iterable[int] | None = None,
              state_path: Path = COLLECTION_STATE_PATH,
              raw_dir: Path = RAW_DATA_DIR, interim_dir: Path = INTERIM_DATA_DIR,
              max_requests: int | None = None, stock_codes: set[str] | None = None,
              max_attempts: int = COLLECTION_MAX_ATTEMPTS,
              retry_seconds: float = COLLECTION_RETRY_SECONDS,
              retry_unavailable: bool = False) -> BatchResult:
    """Collect raw statements or disclosure lists with bounded, restartable retries.

    Only identity-eligible, uniquely DART-linked candidates are requested.
    Industry remains pending. Statements are explicitly unverified snapshots;
    they are not claimed to be the original public filing at prediction T.
    """
    if stage not in {"statements", "disclosures"}:
        raise ValueError("Unknown collection stage")
    if max_attempts < 1 or (max_requests is not None and max_requests < 1):
        raise ValueError("Request and attempt limits must be positive")
    selected_years = list(range(STUDY_START_YEAR, STUDY_END_YEAR + 1) if years is None else years)
    if not selected_years or any(y < STUDY_START_YEAR or y > STUDY_END_YEAR for y in selected_years):
        raise ValueError("Years outside the recorded study scope")
    linked = pd.read_csv(linked_path, dtype="string")
    required = {"corp_code", "stock_code", "collection_status", "corp_code_match_method"}
    if not required.issubset(linked.columns):
        raise ValueError("Rebuild reviewed collection candidates before resuming")
    eligible = linked.loc[linked.collection_status.eq("ready") & linked.corp_code_match_method.isin(["stock_code", "company_name_exact"])]
    targets = eligible[["corp_code", "stock_code"]].dropna().drop_duplicates().sort_values(["stock_code", "corp_code"])
    if targets.stock_code.duplicated().any():
        raise ValueError("Ambiguous DART mapping in collection candidates")
    if stock_codes is not None:
        targets = targets.loc[targets.stock_code.isin(stock_codes)]
        if set(targets.stock_code) != stock_codes:
            raise ValueError("Requested smoke-test stock is not uniquely eligible")
    store = RequestStore(state_path)
    output = raw_dir / f"collection_v2_{stage}_{as_of.isoformat()}_{uuid4().hex[:12]}.csv"
    result = BatchResult(output)
    active_keys: list[str] = []
    stop = False
    try:
        store.ensure_recollection()
        log.line("collection_policy=clean_recollection_no_legacy; old_files_preserved=True")
        log.line(f"stage={stage}; candidate_companies={len(targets)}; raw_only=True")
        for row in targets.itertuples(index=False):
            for year in (selected_years if stage == "statements" else [None]):
                if stage == "statements":
                    key = statement_key(row.corp_code, row.stock_code, year)
                else:
                    # Query semantics, not filename, determine whether a list is complete.
                    key = f"{row.corp_code}|{row.stock_code}|{STUDY_START_YEAR}-01-01|{as_of.isoformat()}|A"
                active_keys.append(key)
                if stage == "statements" and store.status(stage, key) == "failed" and store.original_recovery(key):
                    result.failed += 1
                    result.recovered += 1
                    result.skipped += 1
                    log.line("request_outcome=failed; verified_original_recovery=True; API_retry_skipped=True; source_tables_separate=True")
                    continue
                for attempt in range(max_attempts):
                    status_before = store.status(stage, key)
                    reusable = status_before == "success" or (status_before == "unavailable" and not retry_unavailable)
                    if not reusable and max_requests is not None and result.requests >= max_requests:
                        active_keys.pop()
                        stop = True
                        break

                    def fetch() -> pd.DataFrame:
                        """Fetch and validate one raw response without exposing authentication."""
                        nonlocal client
                        if client is None:
                            client = create_dart_client()
                        if stage == "statements":
                            frame = fetch_annual_statement_with_fallback(client, row.corp_code, year)
                            if not frame.empty:
                                required_fields = {"corp_code", "bsns_year", "reprt_code", "rcept_no", "fs_div_used"}
                                if not required_fields.issubset(frame.columns):
                                    raise ValueError("Unexpected annual statement schema")
                                if not (frame.corp_code.astype(str).eq(row.corp_code).all()
                                        and frame.bsns_year.astype(str).eq(str(year)).all()
                                        and frame.reprt_code.astype(str).eq("11011").all()
                                        and frame.rcept_no.notna().all()):
                                    raise ValueError("Statement request identity mismatch")
                                frame["business_year_requested"] = year
                        else:
                            frame = client.fetch_disclosures(row.corp_code, date(STUDY_START_YEAR, 1, 1), as_of)
                            if not frame.empty:
                                if not {"corp_code", "rcept_no", "rcept_dt", "report_nm"}.issubset(frame.columns):
                                    raise ValueError("Unexpected disclosure schema")
                                if not frame.corp_code.astype(str).eq(row.corp_code).all():
                                    raise ValueError("Disclosure company mismatch")
                        if not frame.empty:
                            frame["stock_code"] = row.stock_code
                        return frame

                    status, skipped, code = store.run(stage, key, fetch, retry_unavailable)
                    result.requests += int(not skipped)
                    result.skipped += int(skipped)
                    if status != "failed" or attempt == max_attempts - 1 or code in {"010", "011", "012", "020", "RuntimeError"}:
                        setattr(result, status, getattr(result, status) + 1)
                        log.line(f"request_outcome={status}; reused={skipped}; attempts_this_batch={attempt + 1}; error_code={code or 'none'}")
                        if code in {"010", "011", "012", "020", "RuntimeError"}:
                            raise CollectionIncomplete(f"collection_stopped_code={code}")
                        break
                    if max_requests is not None and result.requests >= max_requests:
                        result.failed += 1
                        log.line(f"request_outcome=failed; retry_on_restart=True; error_code={code}")
                        break
                    time.sleep(retry_seconds * (2 ** attempt))
                if stop:
                    break
            if stop:
                break
    finally:
        try:
            result.rows = store.export(stage, active_keys, output)
            log.line(f"batch_result success={result.success}; unavailable={result.unavailable}; failed={result.failed}; recovered_from_original={result.recovered}; reused={result.skipped}; new_requests={result.requests}; rows={result.rows}")
        finally:
            store.close()
    return result
