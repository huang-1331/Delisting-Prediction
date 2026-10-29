"""Preserve receipt-specific annual filing ZIPs, including all listed versions."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
import time
from zipfile import ZipFile

import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import RequestStore, file_sha256
from .config import (PROJECT_ROOT, INTERIM_DATA_DIR, RAW_DATA_DIR,
                     MARKET_PROBE_STATE_PATH, get_dart_api_key)
from .data_collection import DartApiClient
from .task03_collect import PacedOpener


def verify_archive(path: Path, digest: str) -> None:
    """Require unchanged bytes and ZIP integrity before reusing a source document."""
    if not path.exists() or file_sha256(path) != digest:
        raise ValueError("Saved source document hash mismatch")
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Saved source document failed CRC")


def main() -> int:
    """Download actual manifest receipts; save metadata and never synthesize reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--http-budget", type=int, default=10000)
    args = parser.parse_args()
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_receipt_archives_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    store = RequestStore(MARKET_PROBE_STATE_PATH)
    opener = PacedOpener(args.http_budget)
    started = time.monotonic()
    result = {"at_kst": now.isoformat(), "manifest": str(args.manifest),
              "new_success": 0, "unavailable": 0, "failed": 0, "reused": 0,
              "archive_integrity_verified": 0, "task03_complete": False,
              "first_public_values_verified": False, "archive_bytes": 0}
    keys = []
    try:
        manifest = pd.read_csv(args.manifest, dtype="string")
        receipts = manifest.rcept_no.dropna().drop_duplicates().tolist()
        result["manifest_receipts"] = len(receipts)
        if not all(r.isdigit() and len(r) == 14 for r in receipts):
            raise ValueError("Manifest receipt identity invalid")
        client = DartApiClient(get_dart_api_key(required=True), opener=opener)
        for index, receipt in enumerate(receipts, 1):
            path = RAW_DATA_DIR / f"receipt_original_{receipt}.zip"

            def fetch() -> pd.DataFrame:
                """Save the received version intact; never overwrite an earlier archive."""
                payload = client.fetch_document_archive(receipt)
                with ZipFile(BytesIO(payload)) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("Received source document failed CRC")
                if path.exists():
                    if path.read_bytes() != payload:
                        alternative = RAW_DATA_DIR / f"receipt_conflict_{receipt}_{stamp}.zip"
                        alternative.write_bytes(payload)
                        raise ValueError("Receipt bytes changed; preserve both for review")
                else:
                    temporary = path.with_suffix(f".{stamp}.tmp")
                    temporary.write_bytes(payload)
                    temporary.replace(path)
                return pd.DataFrame({"receipt_no": [receipt],
                    "path": [path.relative_to(PROJECT_ROOT).as_posix()],
                    "sha256": [file_sha256(path)], "source": ["OpenDART document.xml"],
                    "collected_at_utc": [datetime.now(timezone.utc).isoformat()]})

            keys.append(receipt)
            for attempt in range(3):
                status, reused, code = store.run("dart_receipt_original", receipt, fetch)
                if status != "failed" or code not in {"TimeoutError", "URLError", "HTTPError", "ConnectionResetError"} or attempt == 2:
                    break
                log.line(f"archive_request_retry=True; attempt={attempt + 1}; error_type={code}")
                time.sleep(2 ** attempt)
            result["reused"] += int(reused)
            if not reused:
                result[{"success": "new_success", "unavailable": "unavailable", "failed": "failed"}[status]] += 1
            if status == "success":
                metadata = store.frame("dart_receipt_original", receipt).iloc[0]
                verify_archive(path, metadata.sha256)
                result["archive_integrity_verified"] += 1
                result["archive_bytes"] += path.stat().st_size
            if status == "failed":
                result["stop_code"] = code
                raise RuntimeError("Archive request failed; inspect status before resuming")
            result["processed_receipts"] = index
            if index % 50 == 0 or index == len(receipts):
                result["http_calls"] = opener.calls
                result["elapsed_seconds"] = round(time.monotonic() - started, 3)
                report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                log.line(f"receipts={index}/{len(receipts)}; new_success={result['new_success']}; reused={result['reused']}; unavailable={result['unavailable']}; failed={result['failed']}; physical_http_calls={opener.calls}", timestamp=True)
        result["manifest_processed"] = True
        return 0
    except Exception as exc:
        result["stop_reason"] = type(exc).__name__
        log.line(f"archive_collection_stopped_type={type(exc).__name__}; code={result.get('stop_code','none')}")
        return 1
    finally:
        output = INTERIM_DATA_DIR / f"receipt_archive_metadata_{stamp}.csv"
        store.export("dart_receipt_original", keys, output)
        result["metadata_export"] = str(output.relative_to(PROJECT_ROOT))
        store.close()
        result["http_calls"] = opener.calls
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.line(f"archive_integrity_verified={result['archive_integrity_verified']}; first_public_values_verified=False; elapsed_seconds={result['elapsed_seconds']}; report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
