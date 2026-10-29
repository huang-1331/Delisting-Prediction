"""Recover receipt-specific source tables without manufacturing API records."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

from lxml import html
import pandas as pd

from .collection_state import RequestStore, statement_key, file_sha256
from .collection_runner import ProgressLog
from .config import (PROJECT_ROOT, RAW_DATA_DIR, INTERIM_DATA_DIR,
                     COLLECTION_STATE_PATH, MARKET_PROBE_STATE_PATH, EVENT_OBSERVATION_END)


def native_number(value: str) -> Decimal | None:
    """Read explicit source numbers for balance checks; blanks/dashes stay missing."""
    value = value.strip().replace(",", "")
    if not value or value == "-":
        return None
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError("Source balance total is not finite")
    return number


def extract_source_tables(payload: bytes, receipt: str, year: int) -> dict:
    """Preserve all cells and verified period columns, retaining scope and source units.

    No accounting account ID or Altman mapping is inferred. Equity-change grids
    are preserved as cells rather than flattened using an assumed year column.
    """
    try:
        decoded = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        decoded = payload.decode("cp949", errors="strict")
    decoded = re.sub(r"^\s*<\?xml[^>]*\?>", "", decoded, count=1)
    tree = html.fromstring(decoded, parser=html.HTMLParser(no_network=True))
    cells, values, groups, balances = [], [], [], []
    for group in tree.xpath("//table-group"):
        code = group.get("aclass", "")
        match = re.fullmatch(r"\{XBRL\}(BS|IS|CF|EF)(?:_([CS]))?\d*", code)
        if not match:
            continue
        kind = match.group(1)
        if match.group(2):
            scope = {"C": "CFS", "S": "OFS"}[match.group(2)]
        else:
            scope = None
            for ancestor in group.iterancestors():
                headings = ancestor.xpath("./title")
                if not headings:
                    continue
                heading = re.sub(r"\s+", "", " ".join(headings[0].itertext()))
                if re.fullmatch(r"(?:\d+\.)?연결재무제표", heading):
                    scope = "CFS"
                elif re.fullmatch(r"(?:\d+\.)?재무제표", heading):
                    scope = "OFS"
                break
            if scope is None:
                raise ValueError("Unsuffixed source table scope is not explicit")
        tables = group.xpath("./table")
        if len(tables) != 2:
            raise ValueError("Financial group does not contain caption and values tables")
        caption = " ".join(" ".join(tables[0].itertext()).split())
        period_parts = re.findall(r"제\s*(\d+)\s*기\s*(.*?)(?=제\s*\d+\s*기|\(단위|$)", caption)
        periods = []
        for term, text in period_parts:
            dates = re.findall(r"\d{4}\.\d{2}\.\d{2}", text)
            if not dates:
                raise ValueError("Financial caption does not identify its period")
            periods.append({"term": int(term), "dates": dates, "end_year": int(dates[-1][:4])})
        if not periods or periods[0]["end_year"] != year:
            raise ValueError("Original table year does not match recovery request")
        unit = re.search(r"\(단위\s*:\s*([^\)]+)\)", caption)
        if unit is None:
            raise ValueError("Original table unit missing")
        unit = unit.group(1).strip()
        raw_rows = []
        for ri, row in enumerate(tables[1].xpath(".//tr")):
            entries = []
            for ci, cell in enumerate(row):
                if not isinstance(cell.tag, str):
                    continue
                text = " ".join(" ".join(cell.itertext()).split())
                entries.append(text)
                cells.append({"receipt_no": receipt, "scope": scope, "statement_type": kind,
                    "source_group_code": code, "row_index": ri, "cell_index": ci,
                    "text": text, "cell_tag": cell.tag, "colspan": cell.get("colspan", "1"),
                    "rowspan": cell.get("rowspan", "1"), "unit": unit,
                    "xpath": tree.getroottree().getpath(cell)})
            raw_rows.append(entries)
        if kind != "EF":
            expected_header = [p["term"] for p in periods]
            actual_header = [int(x) for x in re.findall(r"제\s*(\d+)\s*기", " ".join(raw_rows[0]))]
            if actual_header != expected_header:
                raise ValueError("Year columns are not established by the original header")
            for ri, row in enumerate(raw_rows[1:], 1):
                if len(row) != len(periods) + 1:
                    raise ValueError("Source row has ambiguous period columns")
                for period, value in zip(periods, row[1:]):
                    values.append({"receipt_no": receipt, "scope": scope,
                        "statement_type": kind, "row_index": ri, "account_label_source": row[0],
                        "period_end_year": period["end_year"], "period_term": period["term"],
                        "amount_source_text": value, "unit": unit,
                        "source_version": "receipt_specific", "canonical_account_mapping": "pending"})
            if kind == "BS":
                for index, period in enumerate(periods, 1):
                    totals = {}
                    for label in ("자산총계", "부채총계", "자본총계"):
                        matches = [row for row in raw_rows[1:] if row[0] == label]
                        if len(matches) != 1:
                            raise ValueError("Original balance totals are not unique")
                        totals[label] = native_number(matches[0][index])
                    valid = all(v is not None for v in totals.values()) and totals["자산총계"] == totals["부채총계"] + totals["자본총계"]
                    balances.append({"scope": scope, "year": period["end_year"],
                        "assets_equal_liabilities_plus_equity": valid, "unit": unit})
                    if not valid:
                        raise ValueError("Original balance equation does not reconcile")
        groups.append({"scope": scope, "statement_type": kind, "source_group_code": code,
            "caption": caption, "periods": periods, "unit": unit,
            "xpath": tree.getroottree().getpath(group)})
    expected = {(scope, kind) for scope in ("CFS", "OFS") for kind in ("BS", "IS", "CF", "EF")}
    observed = {(g["scope"], g["statement_type"]) for g in groups}
    scope_evidence = None
    if len(groups) == 4 and observed == {("OFS", kind) for kind in ("BS", "IS", "CF", "EF")}:
        scopes = [n.getparent() for n in tree.xpath("//title")
                  if re.fullmatch(r"(?:\d+\.)?연결재무제표", re.sub(r"\s+", "", " ".join(n.itertext()))) ]
        if len(scopes) == 1:
            note = re.sub(r"\s+", " ", " ".join(scopes[0].itertext())).strip()
            compact = re.sub(r"\s+", "", note).replace("&cr;", "")
            if re.fullmatch(r"(?:\d+\.)?연결재무제표당사는해당사항없음", compact):
                scope_evidence = {"kind": "explicit_CFS_not_applicable", "text": note,
                    "xpath": tree.getroottree().getpath(scopes[0]),
                    "text_sha256": hashlib.sha256(note.encode("utf-8")).hexdigest()}
                expected = observed
    if len(groups) != len(expected) or observed != expected:
        raise ValueError("Expected original financial statement scopes not fully present")
    return {"cells": cells, "period_values": values, "groups": groups, "balance_checks": balances,
            "scope_evidence": scope_evidence}


def verify_recovery_manifest(path: Path, key: str) -> dict:
    """Require a matching verified manifest and unchanged source/export bytes."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("request_key") != key or manifest.get("status") != "verified_original_tables"
            or manifest.get("business_year") != int(key.split("|")[2])):
        raise ValueError("Recovery manifest identity or status not verified")
    if not manifest.get("files") or not manifest.get("sources"):
        raise ValueError("Recovery source provenance missing")
    first = manifest["sources"][0]
    if (first.get("receipt") != manifest.get("original_receipt")
            or first.get("public_date") != manifest.get("T")
            or not re.fullmatch(r"\d{14}", first.get("receipt", ""))):
        raise ValueError("Recovery receipt and first public date do not agree")
    # Public dates come from verified source metadata, not identifier prefixes.
    date.fromisoformat(first["public_date"])
    groups = first.get("groups", [])
    kinds = {"BS", "IS", "CF", "EF"}
    expected = {(scope, kind) for scope in ("CFS", "OFS") for kind in kinds}
    scope_evidence = first.get("scope_evidence")
    if scope_evidence and scope_evidence.get("kind") == "explicit_CFS_not_applicable":
        note = scope_evidence.get("text", "")
        compact = re.sub(r"\s+", "", note).replace("&cr;", "")
        if (not scope_evidence.get("xpath") or not re.fullmatch(r"(?:\d+\.)?연결재무제표당사는해당사항없음", compact)
                or hashlib.sha256(note.encode("utf-8")).hexdigest() != scope_evidence.get("text_sha256")):
            raise ValueError("OFS-only explicit scope evidence missing")
        expected = {("OFS", kind) for kind in kinds}
    if (len(groups) != len(expected) or {(g["scope"], g["statement_type"]) for g in groups} != expected
            or any(g["periods"][0]["end_year"] != manifest["business_year"] for g in groups)):
        raise ValueError("Recovery source table periods not verified")
    for item in manifest["files"]:
        target = (PROJECT_ROOT / item["path"]).resolve()
        if not (target.is_relative_to(RAW_DATA_DIR.resolve()) or target.is_relative_to(INTERIM_DATA_DIR.resolve())):
            raise ValueError("Recovery file outside source/evidence directories")
        if file_sha256(target) != item["sha256"]:
            raise ValueError("Recovery source or export hash changed")
    expected_balances = sum(len(g["periods"]) for g in groups if g["statement_type"] == "BS")
    if len(manifest.get("balance_checks", [])) != expected_balances or not all(x["assets_equal_liabilities_plus_equity"] for x in manifest["balance_checks"]):
        raise ValueError("Recovery balance verification missing")
    return manifest


def main() -> int:
    """Recover the user-approved 2024 case from stored first and corrected receipts."""
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_original_recovery_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    statements, originals = RequestStore(COLLECTION_STATE_PATH), RequestStore(MARKET_PROBE_STATE_PATH)
    key = statement_key("00442455", "082660", 2024)
    result = {"at_kst": now.isoformat(), "request_key": key, "network_requests": 0,
        "api_status_preserved": "failed", "recovery_verified": False, "task03_complete": False}
    try:
        if statements.status("statements", key) != "failed":
            raise ValueError("Expected API failure changed; review before recovery")
        disclosure_key = f"00442455|082660|2015-01-01|{EVENT_OBSERVATION_END}|A"
        filings = statements.frame("disclosures", disclosure_key)
        selected = filings.loc[filings.report_nm.astype(str).eq("사업보고서 (2024.12)")]
        if len(selected) != 1 or str(selected.iloc[0].rcept_no) != "20250324000028":
            raise ValueError("First public receipt is not uniquely established")
        files, sources, decoded = [], [], []
        for receipt in ("20250324000028", "20250409002838"):
            record = originals.frame("dart_receipt_original", receipt)
            public = filings.loc[filings.rcept_no.astype(str).eq(receipt)]
            if len(record) != 1 or len(public) != 1:
                raise ValueError("Receipt metadata or public date missing")
            metadata = record.iloc[0]
            archive_path = PROJECT_ROOT / metadata.path
            if file_sha256(archive_path) != metadata.sha256:
                raise ValueError("Receipt archive hash mismatch")
            with ZipFile(archive_path) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Receipt archive integrity failure")
                parsed = extract_source_tables(archive.read(f"{receipt}.xml"), receipt, 2024)
            public_date = pd.Timestamp(public.iloc[0].rcept_dt).date().isoformat()
            for record in parsed["cells"] + parsed["period_values"]:
                record["public_date_receipt"] = public_date
                record["business_year_requested"] = 2024
                record["receipt_version"] = "first_public" if receipt == "20250324000028" else "later_correction"
                if "period_end_year" in record:
                    record["is_requested_period"] = record["period_end_year"] == 2024
            for suffix, records in (("cells", parsed["cells"]), ("period_values", parsed["period_values"])):
                output = INTERIM_DATA_DIR / f"original_recovery_{receipt}_{suffix}_{stamp}.csv"
                pd.DataFrame(records).to_csv(output, index=False, encoding="utf-8-sig")
                files.append({"path": output.relative_to(PROJECT_ROOT).as_posix(), "sha256": file_sha256(output)})
            files.append({"path": metadata.path, "sha256": metadata.sha256})
            sources.append({"receipt": receipt, "public_date": pd.Timestamp(public.iloc[0].rcept_dt).date().isoformat(),
                "version": "first_public" if receipt == "20250324000028" else "later_correction",
                "groups": parsed["groups"], "raw_cells": len(parsed["cells"]),
                "period_values": len(parsed["period_values"]), "balance_checks": parsed["balance_checks"]})
            decoded.append(parsed)
            log.line(f"source_receipt={receipt}; financial_tables={len(parsed['groups'])}; raw_cells={len(parsed['cells'])}; period_values={len(parsed['period_values'])}; balance_checks={len(parsed['balance_checks'])}; all_balance_checks_passed=True")
        first, correction = decoded
        index = lambda data: {(x["scope"], x["statement_type"], x["row_index"], x["cell_index"]): x["text"] for x in data["cells"]}
        first_cells, correction_cells = index(first), index(correction)
        result["version_comparison"] = {"same_cell_layout": first_cells.keys() == correction_cells.keys(),
            "changed_source_cells_at_same_positions": sum(first_cells[k] != correction_cells[k] for k in first_cells.keys() & correction_cells.keys()),
            "corrected_values_used_as_first_public": False}
        manifest = {"schema_version": 1, "status": "verified_original_tables", "request_key": key,
            "business_year": 2024, "original_receipt": "20250324000028", "T": sources[0]["public_date"],
            "api_request_status": "failed", "first_public_tables_recovered": True,
            "canonical_account_mapping": "pending", "ready_for_model": False,
            "scope_preference": "CFS; OFS source tables also retained",
            "files": files, "sources": sources, "balance_checks": first["balance_checks"]}
        manifest_path = INTERIM_DATA_DIR / f"original_recovery_manifest_00442455_2024_{stamp}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify_recovery_manifest(manifest_path, key)
        statements.register_original_recovery(key, manifest_path)
        if statements.status("statements", key) != "failed" or not statements.original_recovery(key):
            raise ValueError("API failure or source recovery not preserved separately")
        from .reliable_collection import run_batch
        class NoNetwork:
            """Fail immediately if the committed recovery unexpectedly requests API data."""
            def fetch_financial_statement(self, *args, **kwargs):
                raise AssertionError("Committed original recovery must not call the API")
        restart = run_batch(INTERIM_DATA_DIR / "collection_v2_linked_2026-10-28_8dc723d3e083.csv",
            datetime.fromisoformat(EVENT_OBSERVATION_END).date(), "statements", log,
            client=NoNetwork(), years=[2024], stock_codes={"082660"})
        if (restart.requests, restart.failed, restart.recovered, restart.rows) != (0, 1, 1, 0):
            raise ValueError("Live stored recovery restart did not retain separate outcomes")
        result["restart_check"] = {"new_api_requests": restart.requests,
            "api_failure_retained": restart.failed, "original_recovery_reused": restart.recovered,
            "bad_api_rows_exported": restart.rows}
        result.update(recovery_verified=True, manifest=str(manifest_path.relative_to(PROJECT_ROOT)),
            original_receipt="20250324000028", T=manifest["T"], sources=sources)
        log.line("api_status=failed; original_source_recovery=verified; requested_year_not_coerced=True; canonical_account_mapping=pending")
        return 0
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        log.line(f"recovery_error_type={type(exc).__name__}")
        return 1
    finally:
        statements.close()
        originals.close()
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.line(f"recovery_verified={result['recovery_verified']}; report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
