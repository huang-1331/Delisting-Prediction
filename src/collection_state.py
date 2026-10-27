"""Transactional request data and outcomes, independent of export filenames."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import sqlite3
from typing import Callable

import pandas as pd

from .config import ANNUAL_REPORT_CODE
from .data_collection import DartApiError


def statement_key(corp: str, stock: str, year: int) -> str:
    """Return the stable key for a company-year annual CFS/OFS-fallback request."""
    return f"{corp}|{stock}|{year}|{ANNUAL_REPORT_CODE}|CFS_then_OFS"


def file_sha256(path: Path) -> str:
    """Hash a file in chunks without displaying its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RequestStore:
    """Persist response and success/unavailable/failed status in one SQLite commit."""

    def __init__(self, path: Path) -> None:
        """Open a stable state file; no existing CSV or checkpoint is modified."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=120)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS requests (stage TEXT, request_key TEXT, status TEXT CHECK(status IN ('success','unavailable','failed')), attempts INTEGER, payload TEXT, error_code TEXT, provenance TEXT, updated_at TEXT, PRIMARY KEY(stage,request_key))")
        self.db.execute("CREATE TABLE IF NOT EXISTS migrations (path TEXT, sha256 TEXT, PRIMARY KEY(path,sha256))")
        self.db.execute("CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT)")
        self.db.commit()

    def close(self) -> None:
        """Close the database after all committed requests have been preserved."""
        self.db.close()

    def status(self, stage: str, key: str) -> str | None:
        """Return a request's committed outcome, or None when never completed."""
        row = self.db.execute("SELECT status FROM requests WHERE stage=? AND request_key=?", (stage, key)).fetchone()
        return row[0] if row else None

    def ensure_recollection(self) -> None:
        """Enforce choice B: do not treat any imported legacy row as a new success."""
        if self.db.execute("SELECT 1 FROM requests WHERE provenance != 'live_request' LIMIT 1").fetchone():
            raise ValueError("Legacy state cannot be used as a clean recollection store")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES ('collection_policy','clean_recollection_no_legacy')")

    def _put(self, stage: str, key: str, status: str, frame: pd.DataFrame,
             attempts: int, error: str, provenance: str) -> None:
        """Write an outcome inside the caller's transaction; retain response rows."""
        payload = frame.to_json(orient="table", index=False, date_format="iso", force_ascii=False)
        self.db.execute("INSERT OR REPLACE INTO requests VALUES (?,?,?,?,?,?,?,?)",
            (stage, key, status, attempts, payload, error, provenance,
             datetime.now(timezone.utc).isoformat()))

    def run(self, stage: str, key: str, fetch: Callable[[], pd.DataFrame],
            retry_unavailable: bool = False) -> tuple[str, bool, str]:
        """Fetch once or reuse success; commit response and status atomically.

        SQLite serializes concurrent writers through the network call. A crash
        rolls back both response and outcome; restarting cannot append duplicates.
        Exceptions are represented by type/status only, never credential-bearing URLs.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT status,attempts FROM requests WHERE stage=? AND request_key=?", (stage, key)).fetchone()
            if row and (row[0] == "success" or (row[0] == "unavailable" and not retry_unavailable)):
                self.db.commit()
                return row[0], True, ""
            attempts = (row[1] if row else 0) + 1
            code = ""
            try:
                frame = fetch()
                status = "unavailable" if frame.empty else "success"
            except Exception as exc:
                code = exc.status if isinstance(exc, DartApiError) else type(exc).__name__
                # Do not save str(exc): HTTP errors may contain authentication queries.
                status = "unavailable" if code == "013" else "failed"
                frame = pd.DataFrame()
            self._put(stage, key, status, frame, attempts, code, "live_request")
            self.db.commit()
            return status, False, code
        except BaseException:
            self.db.rollback()
            raise

    def frame(self, stage: str, key: str) -> pd.DataFrame:
        """Read one stored response without reissuing a successful request."""
        row = self.db.execute("SELECT payload FROM requests WHERE stage=? AND request_key=? AND status='success'", (stage, key)).fetchone()
        return pd.read_json(StringIO(row[0]), orient="table") if row else pd.DataFrame()

    def export(self, stage: str, keys: list[str], output: Path) -> int:
        """Atomically export selected successful responses without appending rows."""
        frames = [self.frame(stage, key) for key in dict.fromkeys(keys)]
        frames = [frame for frame in frames if not frame.empty]
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if stage == "statements" and not frame.empty:
            frame["statement_version_status"] = "unverified_api_snapshot"
            frame["point_in_time_verified"] = False
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(output)
        return len(frame)

    def migrate_legacy(self, raw_dir: Path, interim_dir: Path) -> dict[str, int]:
        """Reconcile legacy files by actual payload/evidence; never trust completed alone.

        Original files remain byte-for-byte intact. Conflicting filing versions
        are not accepted as successes. Unknown old completion IDs become retryable.
        Disclosure ranges cannot be inferred from old rows alone and are rechecked.
        """
        if self.db.execute("SELECT 1 FROM settings WHERE name='collection_policy' AND value='clean_recollection_no_legacy'").fetchone():
            raise ValueError("Choice B forbids importing legacy completion status")
        counts = {"files": 0, "success": 0, "unavailable": 0, "failed": 0}
        sources = [(p, "data") for p in sorted(raw_dir.glob("dart_annual_statements_*.csv"))]
        sources += [(p, "issues") for p in sorted(raw_dir.glob("dart_annual_statement_issues_*.csv"))]
        sources += [(p, "checkpoint") for p in sorted(interim_dir.glob("dart_annual_statement_checkpoint_*.json"))]
        for path, kind in sources:
            digest = file_sha256(path)
            if self.db.execute("SELECT 1 FROM migrations WHERE path=? AND sha256=?", (str(path.resolve()), digest)).fetchone():
                continue
            with self.db:
                if kind == "data":
                    data = pd.read_csv(path, dtype="string", encoding="utf-8-sig").drop_duplicates()
                    for (corp, stock, year), group in data.groupby(["corp_code", "stock_code", "business_year_requested"], dropna=False):
                        if pd.isna(corp) or pd.isna(stock) or pd.isna(year):
                            raise ValueError("Legacy data contains missing request identity")
                        key = statement_key(str(corp), str(stock), int(year))
                        if self.status("statements", key) == "success":
                            continue
                        consistent = group["rcept_no"].notna().all() and group["rcept_no"].nunique() == 1 and group["fs_div_used"].nunique() == 1
                        status = "success" if consistent else "failed"
                        self._put("statements", key, status, group if consistent else pd.DataFrame(), 0,
                                  "" if consistent else "legacy_version_conflict", f"{path.name}:{digest}")
                        counts[status] += 1
                elif kind == "issues":
                    data = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
                    for row in data.itertuples(index=False):
                        key = statement_key(str(row.corp_code), str(row.stock_code), int(row.business_year))
                        if self.status("statements", key) in {"success", "unavailable"}:
                            continue
                        status = "unavailable" if row.issue == "No CFS or OFS annual statement returned" else "failed"
                        self._put("statements", key, status, pd.DataFrame(), 0, "legacy_no_data" if status == "unavailable" else "legacy_failed", f"{path.name}:{digest}")
                        counts[status] += 1
                else:
                    payload = json.loads(path.read_text(encoding="utf-8-sig"))
                    for old_key in payload.get("completed", []):
                        corp, stock, year = old_key.split("|")
                        key = statement_key(corp, stock, int(year))
                        if self.status("statements", key) is None:
                            self._put("statements", key, "failed", pd.DataFrame(), 0, "legacy_completion_without_evidence", f"{path.name}:{digest}")
                            counts["failed"] += 1
                self.db.execute("INSERT INTO migrations VALUES (?,?)", (str(path.resolve()), digest))
            counts["files"] += 1
        return counts
