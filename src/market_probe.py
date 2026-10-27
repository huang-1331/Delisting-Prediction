"""Inspect official KRX market-data contracts without exposing credentials."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import time
from urllib.parse import urljoin, urlparse

from .collection_runner import ProgressLog
from .config import PROJECT_ROOT
from .data_collection import KrxDataClient, _read_url_bytes


def main() -> int:
    """Authenticate and inspect public screen scripts; persist only source metadata."""
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    path = PROJECT_ROOT / "reports/monthly" / f"task03_market_discovery_{stamp}.json"
    log = ProgressLog(path.with_suffix(".txt"))
    result = {"at_kst": now.isoformat(), "screens": [], "completed": 0,
              "failed": 0, "unavailable": 0, "market_records_collected": 0}
    started = time.monotonic()
    try:
        client = KrxDataClient()
        client.authenticate()
        result["authenticated"] = True
        for screen in ("MDCSTAT015", "MDCSTAT017"):
            url = f"https://data.krx.co.kr/contents/MDC/STAT/standard/{screen}.jsp"
            payload = _read_url_bytes(url, opener=client._opener,
                headers={"Referer": "https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd"})
            html = payload.decode("utf-8", errors="replace")
            endpoints = sorted(set(re.findall(r"dbms/MDC/STAT/[A-Za-z0-9_/]+", html)))
            scripts = sorted(set(urljoin(url, item) for item in re.findall(r'<script[^>]+src=[\"\']([^\"\']+)', html)))
            scripts = [item for item in scripts if urlparse(item).hostname == "data.krx.co.kr"]
            result["screens"].append({"url": url, "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(), "data_endpoints": endpoints,
                "official_scripts": scripts,
                "has_market_cap_field": "MKTCAP" in html,
                "has_share_count_field": "LIST_SHRS" in html,
                "has_trade_date_parameter": "trdDd" in html,
                "has_issue_parameter": "isuCd" in html,
                "input_parameters": sorted(set(re.findall(r'name=[\"\']([A-Za-z][A-Za-z0-9_]*)[\"\']', html))),
                "output_field_names": sorted(set(re.findall(r'\b(?:ISU_[A-Z_]+|TDD_[A-Z_]+|MKT_[A-Z_]+|ACC_[A-Z_]+|LIST_SHRS|MKTCAP|SECUGRP_NM|SUSPNT_[A-Z_]+)\b', html))),
                "mentions_suspension": "거래정지" in html})
            result["completed"] += 1
            time.sleep(0.5)
    except Exception as exc:
        result["failed"] += 1
        result["failure_type"] = type(exc).__name__
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["log"] = path.with_suffix(".txt").relative_to(PROJECT_ROOT).as_posix()
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        log.close()
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
