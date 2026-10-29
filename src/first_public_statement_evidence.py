"""Extract first-public tables only for explicitly reviewed manufacturing T rows."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from zipfile import ZipFile

import pandas as pd

from .config import PROJECT_ROOT, INTERIM_DATA_DIR, RAW_DATA_DIR
from .collection_runner import ProgressLog
from .collection_state import file_sha256, statement_key
from .industry_evidence import read_receipt
from .statement_recovery import extract_source_tables, verify_recovery_manifest


def main() -> int:
    """Preserve verified native tables separately from API snapshots and account mapping."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-report", required=True)
    args = parser.parse_args()
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_first_public_tables_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    state = sqlite3.connect(INTERIM_DATA_DIR / "first_public_statement_evidence_v1.sqlite3")
    state.execute("CREATE TABLE IF NOT EXISTS source_tables (receipt TEXT PRIMARY KEY, source_sha256 TEXT, manifest_path TEXT, manifest_sha256 TEXT)")
    result = {"at_kst": now.isoformat(), "network_requests": 0, "new_success": 0,
        "reused": 0, "failed": 0, "api_request_status_changed": False,
        "ready_for_model": False, "canonical_account_mapping": "pending",
        "task03_complete": False, "task04_executed": False, "manifests": []}
    try:
        review_path = PROJECT_ROOT / args.review_report
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("stop_error_type"):
            raise ValueError("Incomplete source review report")
        reviewed_path = PROJECT_ROOT / review["industry_artifact"]
        if file_sha256(reviewed_path) != review["industry_sha256"]:
            raise ValueError("Reviewed industry facts changed")
        reviewed = pd.read_csv(reviewed_path, dtype="string")
        eligible = reviewed.loc[reviewed.manufacturing_confirmed.eq("1")]
        all_versions_path = INTERIM_DATA_DIR / "annual_receipt_versions_20261028_221412_902090.csv"
        versions = pd.read_csv(all_versions_path, dtype="string")
        result["reviewed_manufacturing_T_rows"] = len(eligible)
        for row in eligible.itertuples(index=False):
            receipt, year = row.receipt_no, int(row.fiscal_year)
            result["current_receipt"] = receipt
            if row.review_status != "reviewed_original_business_statement":
                raise ValueError("Manufacturing observation is not explicitly reviewed")
            selected = versions.loc[versions.corp_code.eq(row.corp_code)
                & versions.stock_code.eq(row.stock_code) & versions.fiscal_year_reported.eq(str(year))]
            first = selected.loc[selected.version_kind.eq("initial_candidate")]
            if len(first) != 1 or first.iloc[0].rcept_no != receipt or first.iloc[0].rcept_dt != row.T:
                raise ValueError("First-public receipt metadata ambiguous")
            if (pd.to_datetime(selected.rcept_dt) < pd.Timestamp(row.T)).any():
                raise ValueError("Earlier report version needs review")
            path = PROJECT_ROOT / row.source_path
            if not path.resolve().is_relative_to(RAW_DATA_DIR.resolve()) or file_sha256(path) != row.source_sha256:
                raise ValueError("First source bytes changed")
            key = statement_key(row.corp_code, row.stock_code, year)
            old = state.execute("SELECT source_sha256,manifest_path,manifest_sha256 FROM source_tables WHERE receipt=?", (receipt,)).fetchone()
            if old and old[0] == row.source_sha256:
                saved = PROJECT_ROOT / old[1]
                if file_sha256(saved) != old[2]:
                    raise ValueError("First-source manifest changed")
                verified = verify_recovery_manifest(saved, key)
                if verified["T"] != row.T:
                    raise ValueError("First-source public date changed; review required")
                result["reused"] += 1
                result["manifests"].append(old[1])
                log.line(f"first_public_receipt={receipt}; reused=True; API_unchanged=True")
                continue
            identity = read_receipt(path, receipt, row.corp_code)
            with ZipFile(path) as archive:
                raw = archive.read(identity["member"])
            parsed = extract_source_tables(raw, receipt, year)
            for record in parsed["cells"] + parsed["period_values"]:
                record["public_date_receipt"] = row.T
                record["business_year_requested"] = year
                record["receipt_version"] = "verified_first_public"
                if "period_end_year" in record:
                    record["is_requested_period"] = record["period_end_year"] == year
            cells = INTERIM_DATA_DIR / f"first_public_cells_{receipt}_{stamp}.csv"
            values = INTERIM_DATA_DIR / f"first_public_period_values_{receipt}_{stamp}.csv"
            pd.DataFrame(parsed["cells"]).to_csv(cells, index=False, encoding="utf-8-sig")
            pd.DataFrame(parsed["period_values"]).to_csv(values, index=False, encoding="utf-8-sig")
            manifest = {"status": "verified_original_tables", "request_key": key,
                "business_year": year, "original_receipt": receipt, "T": row.T,
                "files": [{"path": str(p.relative_to(PROJECT_ROOT)), "sha256": file_sha256(p)}
                          for p in (path, cells, values, all_versions_path, reviewed_path)],
                "sources": [{"receipt": receipt, "public_date": row.T,
                    "groups": parsed["groups"], "document": identity["member"],
                    "decoded_encoding": identity["decoded_encoding"]}],
                "balance_checks": parsed["balance_checks"],
                "source_scope_policy": "CFS preferred; OFS preserved separately, no mixing",
                "canonical_account_mapping": "pending", "ready_for_model": False,
                "API_status_changed": False}
            target = INTERIM_DATA_DIR / f"first_public_manifest_{receipt}_{stamp}.json"
            target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            verify_recovery_manifest(target, key)
            with state:
                state.execute("INSERT OR REPLACE INTO source_tables VALUES (?,?,?,?)", (receipt,
                    row.source_sha256, str(target.relative_to(PROJECT_ROOT)), file_sha256(target)))
            result["new_success"] += 1
            result["manifests"].append(str(target.relative_to(PROJECT_ROOT)))
            log.line(f"first_public_receipt={receipt}; verified_groups={len(parsed['groups'])}; balance_checks={len(parsed['balance_checks'])}; account_mapping=pending; API_unchanged=True")
        result["database_integrity"] = state.execute("PRAGMA quick_check").fetchone()[0]
        return 0
    except Exception as exc:
        result["failed"] += 1
        result["stop_error_type"] = type(exc).__name__
        log.line(f"first_public_extraction_stopped={type(exc).__name__}; successful_checkpoints_preserved=True")
        return 1
    finally:
        state.close()
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.line(json.dumps(result, ensure_ascii=False))
        log.line(f"report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
