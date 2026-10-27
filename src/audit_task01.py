"""Audit local evidence without network calls, credentials, or model training."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import unittest
from datetime import datetime, timedelta, timezone

from .collection_runner import ProgressLog
from .config import PROJECT_ROOT


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a local CSV for aggregate checks; return rows only in memory."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fingerprint(path: Path) -> dict[str, object]:
    """Return a relative path, byte size and SHA-256 without file contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def audit() -> dict[str, object]:
    """Inspect local artifacts and run offline checks, returning aggregate evidence."""
    import numpy as np
    import pandas as pd
    from .altman import calculate_altman_z_score, classify_altman_zone
    from .data_collection import build_kosdaq_manufacturing_history
    from .evaluation import evaluate_binary_classifier
    from .labeling import classify_delisting_reasons

    result: dict[str, object] = {"python": platform.python_version()}
    packages = {}
    for package in ("pandas", "numpy", "scikit-learn", "xgboost", "opendartreader",
                    "finance-datareader", "matplotlib", "seaborn", "jupyter", "lxml"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result["packages"] = packages
    originals = [path for folder in ("data", "reports", "notebooks")
                 for path in (PROJECT_ROOT / folder).rglob("*")
                 if path.is_file() and path.name != ".gitkeep"
                 and not path.name.startswith("task01_audit_")]
    result["inventory"] = [fingerprint(path) for path in sorted(originals)]
    raw = PROJECT_ROOT / "data/raw"
    interim = PROJECT_ROOT / "data/interim"
    counts = {}
    for path in sorted([*raw.glob("*.csv"), *interim.glob("*.csv")]):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            count = 0
            stock_codes, corp_codes, company_years, receipts = set(), set(), set(), set()
            receipts_missing, market_rows = 0, 0
            checkpoint_ids = set()
            for row in reader:
                count += 1
                if row.get("stock_code"):
                    stock_codes.add(row["stock_code"])
                if row.get("corp_code"):
                    corp_codes.add(row["corp_code"])
                year = row.get("business_year_requested") or row.get("bsns_year")
                if year:
                    company_years.add((row.get("corp_code"), year))
                    checkpoint_ids.add(f"{row.get('corp_code')}|{row.get('stock_code')}|{year}")
                if "rcept_no" in row:
                    if row["rcept_no"]:
                        receipts.add(row["rcept_no"])
                    else:
                        receipts_missing += 1
                if row.get("market_value_equity"):
                    market_rows += 1
            counts[path.relative_to(PROJECT_ROOT).as_posix()] = {
                "rows": count, "columns": columns, "stock_codes": len(stock_codes),
                "corp_codes": len(corp_codes), "company_years": len(company_years),
                "receipts": len(receipts), "missing_receipt_rows": receipts_missing,
                "market_value_equity_nonempty_rows": market_rows}
            if path.name.startswith("dart_annual_statements_"):
                statement_ids = checkpoint_ids
    result["csv_aggregates"] = counts
    checkpoint_path = interim / "dart_annual_statement_checkpoint_2026-08-01.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))["completed"]
    result["checkpoint"] = {"entries": len(checkpoint), "unique_entries": len(set(checkpoint)),
                            "statement_ids_not_checkpointed": len(statement_ids - set(checkpoint))}
    current = pd.read_csv(raw / "kind_kosdaq_listings_2026-08-01.csv", dtype="string")
    delisted = pd.read_csv(raw / "krx_kosdaq_delisted_2015_2026-08-01.csv", dtype="string")
    history = build_kosdaq_manufacturing_history(current, delisted)
    equity = delisted.loc[delisted.security_type.eq("주권")]
    classified = classify_delisting_reasons(equity.delisting_reason)
    result["universe"] = {"rebuilt_history_rows": len(history),
        "rebuilt_delisted_rows": int(history.delisting_date.notna().sum()),
        "raw_delisted_equity_rows": len(equity),
        "equity_missing_industry": int(equity.industry.isna().sum()),
        "equity_reason_distress": int(classified.eq(True).sum()),
        "equity_reason_competing": int(classified.eq(False).sum()),
        "equity_reason_unknown": int(classified.isna().sum()),
        "unknown_distinct_reasons": int(equity.loc[classified.isna(), "delisting_reason"].nunique(dropna=False))}
    codes = pd.read_csv(raw / "dart_corp_codes_2026-08-01.csv", dtype="string")
    from .data_collection import link_listing_history_to_dart_codes
    linked = link_listing_history_to_dart_codes(equity, codes)
    result["universe"]["equity_dart_linked_rows"] = len(linked.linked)
    result["universe"]["equity_dart_unresolved_rows"] = len(linked.unresolved)
    notebooks = []
    for path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        book = json.loads(path.read_text(encoding="utf-8"))
        code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
        notebooks.append({"path": path.relative_to(PROJECT_ROOT).as_posix(),
            "code_cells": len(code), "execution_count_nonnull": sum(cell.get("execution_count") is not None for cell in code),
            "output_cells": sum(bool(cell.get("outputs")) for cell in code)})
    result["notebooks"] = notebooks
    parsed = []
    for folder in ("src", "tests"):
        for path in sorted((PROJECT_ROOT / folder).glob("*.py")):
            ast.parse(path.read_text(encoding="utf-8-sig"))
            parsed.append(path.relative_to(PROJECT_ROOT).as_posix())
    result["syntax_parsed"] = parsed
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"))
    tests = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    result["offline_tests"] = {"run": tests.testsRun, "failures": len(tests.failures),
                               "errors": len(tests.errors), "passed": tests.wasSuccessful()}
    sample = pd.DataFrame({name: [1.0] for name in (
        "working_capital", "retained_earnings", "ebit", "market_value_equity",
        "sales", "total_assets", "total_liabilities")})
    score = float(calculate_altman_z_score(sample).iloc[0])
    result["synthetic_probes"] = {"altman_unit_ratios_score": score,
        "altman_expected_7_5": bool(np.isclose(score, 7.5)),
        "missing_altman_zone": str(classify_altman_zone(pd.Series([np.nan])).iloc[0]),
        "single_class_confusion_shape": list(evaluate_binary_classifier(
            np.array([0, 0]), np.array([0, 0])).confusion_matrix.shape)}
    old_log = PROJECT_ROOT / "reports/monthly/2026-08_actual_collection_terminal.txt"
    try:
        log_text = old_log.read_text(encoding="utf-8-sig")
        old_log_encoding = "utf-8-sig"
    except UnicodeDecodeError:
        log_text = old_log.read_text(encoding="cp949")
        old_log_encoding = "cp949"
    result["old_terminal_log"] = {"path": old_log.relative_to(PROJECT_ROOT).as_posix(),
        "line_count": len(log_text.splitlines()), "encoding": old_log_encoding,
        "has_2875_count": "2,875" in log_text or "2875" in log_text,
        "has_475044_count": "475,044" in log_text or "475044" in log_text,
        "has_interrupted_marker": "INTERRUPTED" in log_text,
        "has_incomplete_marker": "INCOMPLETE" in log_text,
        "direct_tee_provenance": "not established by file existence or contents"}
    result["missing_artifact_categories"] = {
        "disclosure_csv_count": len(list(raw.glob("*disclosure*.csv"))),
        "processed_files": len([p for p in (PROJECT_ROOT / "data/processed").rglob("*") if p.is_file() and p.name != ".gitkeep"]),
        "market_named_files": len([p for p in originals if any(t in p.name.lower() for t in ("market_value", "market_cap", "price", "시가총액"))])}
    return result


def main() -> int:
    """Tee the actual audit execution and save only aggregate JSON evidence."""
    now = datetime.now(timezone(timedelta(hours=9)))
    prefix = f"task01_audit_{now:%Y%m%d_%H%M%S}"
    report_dir = PROJECT_ROOT / "reports/monthly"
    log = ProgressLog(report_dir / f"{prefix}.txt")
    try:
        print(f"Task 01 offline audit started {now.isoformat()}", flush=True)
        result = audit()
        result["audited_at_kst"] = now.isoformat()
        result["execution_log"] = f"reports/monthly/{prefix}.txt"
        result_path = report_dir / f"{prefix}.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print(f"Aggregate evidence saved: {result_path.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
        return 0 if result["offline_tests"]["passed"] else 1
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
