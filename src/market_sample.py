"""Bounded official-source market-value feasibility evidence, not model inputs."""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import time

import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import RequestStore, file_sha256
from .config import (PROJECT_ROOT, COLLECTION_STATE_PATH, MARKET_PROBE_STATE_PATH,
                     RAW_DATA_DIR, INTERIM_DATA_DIR)
from .data_collection import create_dart_client, KrxDataClient


def market_number(value: object) -> Decimal | None:
    """Parse an explicit source number; missing/invalid entries remain missing."""
    if value is None or pd.isna(value):
        return None
    try:
        number = Decimal(str(value).replace(",", "").strip())
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def annual_anchor(disclosures: pd.DataFrame, year: int) -> pd.Series | None:
    """Select the earliest listed, non-correction December annual filing for a pilot.

    Receipt-list evidence only; this does not verify the original statement body
    or establish final study eligibility. Other fiscal year-ends are not guessed.
    """
    if disclosures.empty:
        return None
    names = disclosures.report_nm.astype("string")
    matches = disclosures.loc[names.str.contains("사업보고서", na=False)
        & names.str.contains(rf"\({year}\.12\)", regex=True, na=False)
        & ~names.str.contains("정정|철회", regex=True, na=False)].copy()
    if matches.empty:
        return None
    matches["rcept_dt"] = pd.to_datetime(matches.rcept_dt)
    return matches.sort_values(["rcept_dt", "rcept_no"]).iloc[0]


def main() -> int:
    """Probe four technical cases with cached receipts and daily official snapshots."""
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report_path = PROJECT_ROOT / "reports/monthly" / f"task03_market_sample_{stamp}.json"
    log = ProgressLog(report_path.with_suffix(".txt"))
    result = {"at_kst": now.isoformat(), "scope": "Altman feasibility pilot only",
              "completed_requests": 0, "unavailable_requests": 0, "failed_requests": 0,
              "reused_requests": 0, "unresolved_cases": 0, "cases": []}
    started = time.monotonic()
    source_files = [*RAW_DATA_DIR.glob("dart_*.csv"), *RAW_DATA_DIR.glob("kind_*.csv"),
                    *RAW_DATA_DIR.glob("krx_*.csv"), *INTERIM_DATA_DIR.glob("dart_*checkpoint*.json")]
    before = {path: file_sha256(path) for path in source_files}
    financial = RequestStore(COLLECTION_STATE_PATH)
    market = RequestStore(MARKET_PROBE_STATE_PATH)
    keys_used = []
    case_manifest = []
    try:
        smoke = json.loads((PROJECT_ROOT / "reports/monthly/task02_smoke_20261028_003448_132704.json").read_text(encoding="utf-8"))
        candidates = pd.read_csv(PROJECT_ROOT / smoke["candidate_artifacts"]["linked"], dtype="string")
        eligible = candidates.loc[candidates.collection_status.eq("ready")].copy()
        old_keys = financial.db.execute("SELECT request_key FROM requests WHERE stage='statements' AND status='success' ORDER BY request_key").fetchall()
        samples = []
        for (key,) in old_keys[:2]:
            corp, stock, _, _, _ = key.split("|")
            row = eligible.loc[eligible.corp_code.eq(corp) & eligible.stock_code.eq(stock)].iloc[0]
            samples.append(("historical_delisted", row, 2022))
        normal = eligible.loc[eligible.delisting_date.isna()
            & (pd.to_datetime(eligible.listing_date) < pd.Timestamp("2022-01-01"))
            & eligible.industry.str.contains("제조업", na=False)].sort_values("stock_code")
        if not normal.empty:
            samples.append(("current_listing_comparison", normal.iloc[0], 2022))
        # This is only a candidate: zero volume alone must never prove suspension.
        suspension = eligible.loc[eligible.stock_code.eq("024810")]
        if not suspension.empty:
            samples.append(("suspension_candidate_unverified", suspension.iloc[0], 2023))
        dart = None
        krx = KrxDataClient()
        for ordinal, (category, row, year) in enumerate(samples, 1):
            case = {"case": ordinal, "category": category, "financial_year": year,
                    "suspension_verified": False, "original_statement_verified": False,
                    "full_equity_verified": False}
            result["cases"].append(case)
            disclosure_key = f"{row.corp_code}|{row.stock_code}|2015-01-01|2026-10-28|A"
            def fetch_disclosures():
                """Fetch source-stamped disclosure rows using the existing request cache."""
                nonlocal dart
                if dart is None:
                    dart = create_dart_client()
                frame = dart.fetch_disclosures(row.corp_code, date(2015, 1, 1), date(2026, 10, 28))
                if not frame.empty:
                    frame["stock_code"] = row.stock_code
                return frame
            status, reused, error = financial.run("disclosures", disclosure_key, fetch_disclosures)
            result["reused_requests"] += int(reused)
            if not reused:
                result[{"success": "completed_requests", "unavailable": "unavailable_requests", "failed": "failed_requests"}[status]] += 1
            if status != "success":
                case["status"] = f"disclosure_{status}"
                case["error_type"] = error
                result["unresolved_cases"] += 1
                continue
            receipt = annual_anchor(financial.frame("disclosures", disclosure_key), year)
            if receipt is None:
                case["status"] = "annual_anchor_unresolved"
                result["unresolved_cases"] += 1
                continue
            anchor = pd.Timestamp(receipt.rcept_dt).date()
            if anchor < pd.Timestamp(row.listing_date).date() or (pd.notna(row.delisting_date) and anchor >= pd.Timestamp(row.delisting_date).date()):
                case["status"] = "anchor_outside_listing"
                result["unresolved_cases"] += 1
                continue
            case_manifest.append({"case": ordinal, "stock_code": row.stock_code,
                "corp_code": row.corp_code, "annual_year": year,
                "rcept_no": str(receipt.rcept_no), "observation_date_candidate": anchor.isoformat(),
                "original_statement_verified": False})
            daily = pd.DataFrame()
            for delta in range(1, 8):
                trade_date = anchor - timedelta(days=delta)
                market_key = f"KSQ|{trade_date.isoformat()}|MDCSTAT01501|native_units"
                keys_used.append(market_key)
                outcome, reused, error = market.run("krx_daily_market", market_key,
                    lambda: krx.fetch_daily_market(trade_date))
                result["reused_requests"] += int(reused)
                if not reused:
                    result[{"success": "completed_requests", "unavailable": "unavailable_requests", "failed": "failed_requests"}[outcome]] += 1
                    time.sleep(0.5)
                if outcome == "failed":
                    case["status"] = "market_request_failed"
                    case["error_type"] = error
                    break
                if outcome == "success":
                    daily = market.frame("krx_daily_market", market_key)
                    break
            if daily.empty:
                case.setdefault("status", "no_market_day_within_pilot_search")
                result["unresolved_cases"] += 1
                continue
            case["observation_year"] = anchor.year
            case["strictly_before_T"] = trade_date < anchor
            selected = daily.loc[daily.ISU_SRT_CD.astype(str).eq(row.stock_code)]
            if len(selected) != 1:
                case["status"] = "issue_missing_or_ambiguous"
                result["unresolved_cases"] += 1
                continue
            record = selected.iloc[0]
            price, shares, value, volume = [market_number(record[field]) for field in
                ("TDD_CLSPRC", "LIST_SHRS", "MKTCAP", "ACC_TRDVOL")]
            case["required_fields_present"] = all(number is not None for number in (price, shares, value, volume))
            case["price_times_shares_matches_market_cap"] = bool(price is not None and shares is not None and value is not None and price * shares == value)
            case["positive_trade_observed"] = bool(volume is not None and volume > 0)
            case["reported_price_positive"] = bool(price is not None and price > 0)
            case["status"] = "source_fields_obtained" if case["required_fields_present"] else "source_fields_missing"
            case["ordinary_issue_only"] = True
            if not case["required_fields_present"]:
                result["unresolved_cases"] += 1
            # Investigate any zero-volume case through separate official evidence.
            if not case["positive_trade_observed"]:
                case["freshness_status"] = "no_trade_observed_do_not_assume_fresh_or_suspended"
            print(f"case={ordinal}; category={category}; outcome={case['status']}; fields_match={case['price_times_shares_matches_market_cap']}; positive_trade={case['positive_trade_observed']}", flush=True)
        result["observation_year_coverage"] = dict(Counter(str(case["observation_year"]) for case in result["cases"] if "observation_year" in case))
        manifest_path = INTERIM_DATA_DIR / f"market_pilot_manifest_{stamp}.json"
        manifest_path.write_text(json.dumps(case_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        result["case_manifest"] = manifest_path.relative_to(PROJECT_ROOT).as_posix()
        raw_path = RAW_DATA_DIR / f"market_pilot_snapshots_{stamp}.csv"
        result["market_snapshot_rows"] = market.export("krx_daily_market", keys_used, raw_path)
        result["market_snapshot_path"] = raw_path.relative_to(PROJECT_ROOT).as_posix()
        result["full_research_collection_started"] = False
        result["full_equity_and_suspension_coverage_verified"] = False
    except Exception as exc:
        result["failure_type"] = type(exc).__name__
        result["failed_requests"] += 1
    finally:
        financial.close()
        market.close()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["original_sources_unchanged"] = all(file_sha256(path) == digest for path, digest in before.items())
        result["pilot_execution_completed"] = "failure_type" not in result
        result["log"] = report_path.with_suffix(".txt").relative_to(PROJECT_ROOT).as_posix()
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        log.close()
    return 0 if result["pilot_execution_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
