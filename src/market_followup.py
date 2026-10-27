"""Gather original filings and last-observed-trade evidence for the market pilot."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from io import BytesIO
import json
import re
import time
from zipfile import ZipFile

from lxml import html
import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import RequestStore, file_sha256
from .config import PROJECT_ROOT, RAW_DATA_DIR, INTERIM_DATA_DIR, MARKET_PROBE_STATE_PATH
from .data_collection import KrxDataClient, create_dart_client
from .market_sample import market_number


def filing_review_material(archive_path) -> dict:
    """Extract review-only stock tables and suspension paragraphs from a receipt ZIP.

    Original bytes and document names remain available. No manufacturing,
    suspension duration or absence of preferred shares is inferred automatically.
    """
    result = {"documents": [], "stock_tables": [], "suspension_paragraphs": []}
    with ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith((".xml", ".html", ".htm")):
                continue
            document = html.fromstring(archive.read(name), parser=html.HTMLParser(no_network=True))
            result["documents"].append(name)
            for index, table in enumerate(document.xpath("//table")):
                text = re.sub(r"\s+", " ", " ".join(table.itertext())).strip()
                if any(phrase in text for phrase in ("발행할 주식의 총수", "발행한 주식의 총수", "발행주식의 총수", "유통주식수")):
                    rows = [[re.sub(r"\s+", " ", " ".join(cell.itertext())).strip()
                        for cell in row if isinstance(cell.tag, str)] for row in table.xpath(".//tr")]
                    result["stock_tables"].append({"document": name, "table_index": index,
                        "cell_tags": sorted(set(cell.tag for row in table.xpath(".//tr") for cell in row if isinstance(cell.tag, str))), "rows": rows})
            for index, paragraph in enumerate(document.xpath("//p")):
                text = re.sub(r"\s+", " ", " ".join(paragraph.itertext())).strip()
                if "거래정지" in text or "거래 정지" in text:
                    result["suspension_paragraphs"].append({"document": name, "paragraph_index": index, "text": text})
    return result


def main() -> int:
    """Collect a bounded extension of the four existing cases and log aggregates."""
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    path = PROJECT_ROOT / "reports/monthly" / f"task03_market_followup_{stamp}.json"
    log = ProgressLog(path.with_suffix(".txt"))
    store = RequestStore(MARKET_PROBE_STATE_PATH)
    result = {"at_kst": now.isoformat(), "cases": [], "completed_requests": 0,
              "unavailable_requests": 0, "failed_requests": 0, "reused_requests": 0}
    started = time.monotonic()
    history_keys = []
    originals = [*RAW_DATA_DIR.glob("dart_*.csv"), *RAW_DATA_DIR.glob("kind_*.csv"),
                 *RAW_DATA_DIR.glob("krx_*.csv"), *INTERIM_DATA_DIR.glob("dart_*checkpoint*.json")]
    before = {item: file_sha256(item) for item in originals}

    def request(stage, key, fetch):
        """Execute/cache one source request and count outcomes, without exception text."""
        status, reused, code = store.run(stage, key, fetch)
        result["reused_requests"] += int(reused)
        if not reused:
            result[{"success": "completed_requests", "unavailable": "unavailable_requests", "failed": "failed_requests"}[status]] += 1
            time.sleep(0.5)
        return status, code

    try:
        cases = json.loads((INTERIM_DATA_DIR / "market_pilot_manifest_20261028_004741_026994.json").read_text(encoding="utf-8"))
        snapshots = pd.read_csv(RAW_DATA_DIR / "market_pilot_snapshots_20261028_004741_026994.csv", dtype="string")
        krx = KrxDataClient()
        dart = None
        for case in cases:
            entry = {"case": case["case"], "original_receipt_archive_verified": False,
                     "suspension_status": "manual_review_required", "full_equity_status": "manual_review_required"}
            result["cases"].append(entry)
            anchor = pd.Timestamp(case["observation_date_candidate"])
            matching = snapshots.loc[snapshots.ISU_SRT_CD.eq(case["stock_code"])
                & (pd.to_datetime(snapshots.requested_trade_date) < anchor)]
            if matching.empty:
                entry["history_status"] = "source_isin_unresolved"
                continue
            matching = matching.sort_values("requested_trade_date")
            latest = matching.iloc[-1]
            isin = str(latest.ISU_CD)
            end = pd.Timestamp(latest.requested_trade_date).date()
            history = pd.DataFrame()
            for lookback in (2, 4, 8):
                start = max(date(2015, 1, 1), (anchor - pd.DateOffset(years=lookback)).date())
                key = f"{isin}|{start.isoformat()}|{end.isoformat()}|MDCSTAT01701|source_default"
                history_keys.append(key)
                if store.status("krx_price_history", key) == "failed":
                    # Preserve the failed range; retry the same evidence in small calendar chunks.
                    parts = []
                    status = "success"
                    for calendar_year in range(start.year, end.year + 1):
                        chunk_start = max(start, date(calendar_year, 1, 1))
                        chunk_end = min(end, date(calendar_year, 12, 31))
                        chunk_key = f"{isin}|{chunk_start.isoformat()}|{chunk_end.isoformat()}|MDCSTAT01701|source_default"
                        history_keys.append(chunk_key)
                        status, code = request("krx_price_history", chunk_key,
                            lambda: krx.fetch_price_history(isin, chunk_start, chunk_end))
                        if status == "failed":
                            break
                        if status == "success":
                            parts.append(store.frame("krx_price_history", chunk_key))
                    history = pd.concat(parts, ignore_index=True) if parts and status != "failed" else pd.DataFrame()
                    entry["failed_range_rechecked_in_year_chunks"] = True
                else:
                    status, code = request("krx_price_history", key,
                        lambda: krx.fetch_price_history(isin, start, end))
                    history = store.frame("krx_price_history", key) if status == "success" else pd.DataFrame()
                entry["history_status"] = status
                if status == "failed" or history.empty:
                    entry["history_error_type"] = code
                    break
                dates = pd.to_datetime(history.TRD_DD, errors="coerce")
                if dates.isna().any() or dates.min().date() < start or dates.max().date() > end:
                    raise ValueError("Historical response date contract failed")
                volumes = history.ACC_TRDVOL.map(market_number)
                traded = history.loc[volumes.map(lambda value: value is not None and value > 0)]
                if not traded.empty:
                    last_index = pd.to_datetime(traded.TRD_DD).idxmax()
                    last_date = pd.Timestamp(history.loc[last_index, "TRD_DD"]).date()
                    entry["last_positive_volume_date_in_covered_history"] = last_date.isoformat()
                    entry["calendar_days_before_T"] = (anchor.date() - last_date).days
                    entry["history_reaches_snapshot_day"] = dates.max().date() == end
                    entry["last_trade_is_snapshot_day"] = last_date == end
                    entry["price_adjustment_status"] = "source_default_unverified"
                    break
            receipt = str(case["rcept_no"])
            archive_path = RAW_DATA_DIR / f"receipt_original_{receipt}.zip"

            def fetch_document():
                """Preserve original receipt bytes; cache only path/hash/source metadata."""
                nonlocal dart
                if dart is None:
                    dart = create_dart_client()
                payload = dart.fetch_document_archive(receipt)
                with ZipFile(BytesIO(payload)) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("Invalid document archive")
                if archive_path.exists():
                    if archive_path.read_bytes() != payload:
                        raise ValueError("Receipt archive changed; retain existing version for review")
                else:
                    with archive_path.open("xb") as stream:
                        stream.write(payload)
                return pd.DataFrame({"receipt_no": [receipt],
                    "path": [archive_path.relative_to(PROJECT_ROOT).as_posix()],
                    "sha256": [file_sha256(archive_path)], "source": ["OpenDART document.xml"],
                    "collected_at_utc": [datetime.now(timezone.utc).isoformat()]})

            status, code = request("dart_receipt_original", receipt, fetch_document)
            entry["document_status"] = status
            if status == "success":
                metadata = store.frame("dart_receipt_original", receipt).iloc[0]
                if not archive_path.exists() or file_sha256(archive_path) != metadata.sha256:
                    raise ValueError("Saved original receipt hash mismatch")
                entry["original_receipt_archive_verified"] = True
                material = filing_review_material(archive_path)
                review_path = INTERIM_DATA_DIR / f"receipt_review_{receipt}_{stamp}.json"
                review_path.write_text(json.dumps(material, ensure_ascii=False, indent=2), encoding="utf-8")
                entry["review_material"] = review_path.relative_to(PROJECT_ROOT).as_posix()
                entry["archive"] = archive_path.relative_to(PROJECT_ROOT).as_posix()
                entry["stock_table_count"] = len(material["stock_tables"])
                entry["suspension_paragraph_count"] = len(material["suspension_paragraphs"])
            else:
                entry["document_error_type"] = code
            print(json.dumps(entry, ensure_ascii=False), flush=True)
        raw = RAW_DATA_DIR / f"market_pilot_histories_{stamp}.csv"
        result["history_rows"] = store.export("krx_price_history", history_keys, raw)
        result["history_path"] = raw.relative_to(PROJECT_ROOT).as_posix()
        result["source_evidence_gathered"] = True
    except Exception as exc:
        result["failure_type"] = type(exc).__name__
        result["source_evidence_gathered"] = False
    finally:
        store.close()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["original_sources_unchanged"] = all(file_sha256(item) == digest for item, digest in before.items())
        result["log"] = path.with_suffix(".txt").relative_to(PROJECT_ROOT).as_posix()
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        log.close()
    return 0 if result["source_evidence_gathered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
