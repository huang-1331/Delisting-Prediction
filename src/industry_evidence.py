"""Receipt-specific industry evidence; extraction never implies eligibility."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from zipfile import ZipFile

from lxml import html
import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import file_sha256
from .config import (PROJECT_ROOT, INTERIM_DATA_DIR, RAW_DATA_DIR,
                     MARKET_PROBE_STATE_PATH, EVENT_OBSERVATION_END,
                     KSIC_EDITION_STARTS, KSIC_MANUFACTURING_DIVISIONS)


def ksic_edition_at_T(value: str) -> int:
    """Return the generally effective edition within this study's public-T range.

    Effective dates: edition 10, 2017-07-01; edition 11, 2024-07-01.
    This is a taxonomy lookup, not a company's historical industry lookup.
    """
    observed = date.fromisoformat(value)
    if not date(2015, 1, 1) <= observed <= date.fromisoformat(EVENT_OBSERVATION_END):
        raise ValueError("T outside the reviewed study interval")
    return max((start, edition) for start, edition in KSIC_EDITION_STARTS
               if observed >= date.fromisoformat(start))[1]


def manufacturing_divisions(edition: int) -> tuple[int, ...]:
    """Return the verified manufacturing division range, not individual code mappings."""
    if edition not in KSIC_MANUFACTURING_DIVISIONS:
        raise ValueError("Unverified KSIC edition")
    first, last = KSIC_MANUFACTURING_DIVISIONS[edition]
    return tuple(range(first, last + 1))


def verify_public_date_page(payload: bytes, receipt: str, expected_date: str,
                            expected_company: str) -> dict:
    """Cross-check official DART title and selected receipt against list metadata.

    A receipt identifier's first eight digits are not treated as a public date.
    """
    tree = html.fromstring(payload.decode("utf-8", errors="strict"))
    titles = tree.xpath("//title")
    options = tree.xpath(f'//option[@selected and @value="rcpNo={receipt}"]')
    if len(titles) != 1 or len(options) != 1:
        raise ValueError("Official page receipt identity ambiguous")
    title = " ".join(titles[0].itertext()).strip()
    option = " ".join(options[0].itertext())
    if title.split("/")[0] != expected_company or "사업보고서" not in title or "사업보고서" not in option:
        raise ValueError("Official page company/report mismatch")
    for text in (title, option):
        dates = re.findall(r"\d{4}\.\d{2}\.\d{2}", text)
        if len(dates) != 1 or dates[0].replace(".", "-") != expected_date:
            raise ValueError("Official page public date disagrees with list")
    return {"receipt": receipt, "public_date": expected_date,
        "title": title, "title_xpath": tree.getroottree().getpath(titles[0]),
        "selected_receipt_xpath": tree.getroottree().getpath(options[0]),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "public_date_crosschecked": True}


def parse_main_document(payload: bytes, corp_code: str):
    """Decode strictly and verify the reporting entity before extracting its text.

    Some original XML files declare UTF-8 but actually contain CP949 bytes.
    No replacement/ignored bytes are allowed; the original archive stays intact.
    """
    try:
        decoded = payload.decode("utf-8-sig", errors="strict")
        encoding = "utf-8"
    except UnicodeDecodeError:
        decoded = payload.decode("cp949", errors="strict")
        encoding = "cp949"
    declared = re.search(r"<\?xml[^>]*encoding=[\"']([^\"']+)", decoded[:200], re.I)
    decoded = re.sub(r"^\s*<\?xml[^>]*\?>", "", decoded, count=1)
    tree = html.fromstring(decoded, parser=html.HTMLParser(no_network=True))
    identities = tree.xpath("//document-header/company-name[@aregcik] | //document/company-name[@aregcik]")
    if len(identities) != 1 or identities[0].get("aregcik") != corp_code:
        raise ValueError("Reporting entity identity unverified")
    names = tree.xpath("//document-name")
    if len(names) != 1 or "사업보고서" not in "".join(names[0].itertext()):
        raise ValueError("Annual report identity unverified")
    return tree, {"decoded_encoding": encoding,
        "declared_encoding": declared.group(1) if declared else None,
        "reporting_name_at_T": " ".join(identities[0].itertext()).strip(),
        "identity_xpath": tree.getroottree().getpath(identities[0])}


def business_evidence(tree) -> list[dict]:
    """Collect overview passages for review; do not classify by keywords.

    Subsidiary financial notes and acquisition-target industry lists are outside
    this selector. Full source paths and XPaths remain available for review.
    """
    compact = lambda node: re.sub(r"\s+", "", "".join(node.itertext()))
    scopes = [n.getparent() for n in tree.xpath("//title")
              if compact(n) in {"1.회사의개요", "1.회사의개황"}]
    if len(scopes) != 1:
        return []
    scope = scopes[0]
    findings = []
    for paragraph in scope.xpath(".//p"):
        if not re.search(r"주요사업(의)?내용|주된사업", compact(paragraph)):
            continue
        nodes = [paragraph]
        # These are extraction candidates, not reviewed company-industry facts.
        for sibling in paragraph.itersiblings():
            text = " ".join(sibling.itertext()).strip()
            if re.match(r"^[가-하]\s*[.)]|^\d+\s*[.)]", text):
                break
            nodes.append(sibling)
            if len(nodes) == 5:
                break
        for node in nodes:
            text = re.sub(r"\s+", " ", " ".join(node.itertext())).strip()
            if text:
                findings.append({"xpath": tree.getroottree().getpath(node),
                    "text": text, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    if not findings:
        # Newer forms move business descriptions out of the company overview.
        # This whole section is review material, including subsidiaries; no
        # company-industry decision may be inferred from its keywords alone.
        alternatives = [n.getparent() for n in tree.xpath("//title")
            if compact(n) == "1.사업의개요"]
        if len(alternatives) == 1:
            node = alternatives[0]
            text = re.sub(r"\s+", " ", " ".join(node.itertext())).strip()
            findings.append({"xpath": tree.getroottree().getpath(node),
                "text": text, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "selection": "business_overview_fallback_requires_entity_scope_review"})
    return findings


def read_receipt(path: Path, receipt: str, corp_code: str) -> dict:
    """Read the exact main filename, including ZIPs with leading-slash names."""
    with ZipFile(path) as archive:
        members = [n for n in archive.namelist()
                   if PurePosixPath(n).name == f"{receipt}.xml"]
        if len(members) != 1:
            raise ValueError("Main source member ambiguous or absent")
        payload = archive.read(members[0])  # ZIP CRC checked by read; no extraction.
    tree, metadata = parse_main_document(payload, corp_code)
    return {**metadata, "member": members[0],
        "member_sha256": hashlib.sha256(payload).hexdigest(),
        "business_passages": business_evidence(tree)}


def main() -> int:
    """Build a resumable historical-first review queue from verified original ZIPs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-only", action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_industry_evidence_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    state_path = INTERIM_DATA_DIR / "industry_evidence_v1.sqlite3"
    state = sqlite3.connect(state_path)
    state.execute("CREATE TABLE IF NOT EXISTS evidence (receipt TEXT PRIMARY KEY, source_sha256 TEXT, status TEXT, payload TEXT, error_type TEXT)")
    archives = sqlite3.connect(f"file:{MARKET_PROBE_STATE_PATH.as_posix()}?mode=ro", uri=True)
    result = {"at_kst": now.isoformat(), "network_requests": 0, "task03_complete": False,
              "task04_executed": False, "new_industry_decisions": 0,
              "new_financial_reconstructions": 0, "historical_only": args.historical_only,
              "new_extractions": 0, "reused": 0, "extraction_failed": 0}
    try:
        manifest_path = INTERIM_DATA_DIR / "annual_receipt_versions_20261028_221412_902090.csv"
        linked_path = INTERIM_DATA_DIR / "collection_v2_linked_2026-10-28_8dc723d3e083.csv"
        manifest = pd.read_csv(manifest_path, dtype="string")
        linked = pd.read_csv(linked_path, dtype="string")
        initial = manifest.loc[manifest.version_kind.eq("initial_candidate")].copy()
        if initial.duplicated(["corp_code", "stock_code", "fiscal_year_reported", "fiscal_month_reported"]).any():
            raise ValueError("Multiple initial-title candidates need review")
        if linked.stock_code.duplicated().any():
            raise ValueError("Ambiguous company identity")
        identities = linked.set_index("stock_code")
        initial["historical"] = initial.stock_code.map(identities.delisting_date.notna())
        if args.historical_only:
            initial = initial.loc[initial.historical]
        initial = initial.sort_values(["historical", "stock_code", "rcept_dt"], ascending=[False, True, True])
        metadata = {key: json.loads(payload)["data"][0]
            for key, payload in archives.execute("SELECT request_key,payload FROM requests WHERE stage='dart_receipt_original' AND status='success'")}
        rows = []
        result["selected_receipts"] = len(initial)
        result["manifest"] = str(manifest_path.relative_to(PROJECT_ROOT))
        result["manifest_sha256"] = file_sha256(manifest_path)
        for index, row in enumerate(initial.itertuples(index=False), 1):
            result["current_receipt"] = row.rcept_no
            identity = identities.loc[row.stock_code]
            public_date = pd.Timestamp(row.rcept_dt).date()
            if row.corp_code != identity.corp_code:
                raise ValueError("Company identity mismatch")
            if not re.fullmatch(r"\d{14}", row.rcept_no):
                raise ValueError("Invalid receipt identifier")
            listing = pd.Timestamp(identity.listing_date).date() if pd.notna(identity.listing_date) else None
            delisting = pd.Timestamp(identity.delisting_date).date() if pd.notna(identity.delisting_date) else None
            interval = None if listing is None else listing <= public_date and (delisting is None or public_date < delisting)
            base = {"corp_code": row.corp_code, "stock_code": row.stock_code,
                "receipt": row.rcept_no, "T": public_date.isoformat(),
                "fiscal_year": row.fiscal_year_reported, "fiscal_month": row.fiscal_month_reported,
                "ksic_edition": ksic_edition_at_T(public_date.isoformat()),
                "public_date_source": "OpenDART list.json rcept_dt",
                "receipt_prefix_differs_from_public_date": row.rcept_no[:8] != public_date.strftime("%Y%m%d"),
                "within_listing_interval": interval,
                "manufacturing_confirmed": None, "industry_review_status": "pending_source_review"}
            source = metadata.get(row.rcept_no)
            if not source:
                base["extraction_status"] = "source_unavailable"
                rows.append(base)
                continue
            path = PROJECT_ROOT / source["path"]
            if not path.resolve().is_relative_to(RAW_DATA_DIR.resolve()) or file_sha256(path) != source["sha256"]:
                raise ValueError("Original source provenance changed")
            base.update({"archive": source["path"], "archive_sha256": source["sha256"]})
            old = state.execute("SELECT source_sha256,status,payload FROM evidence WHERE receipt=?", (row.rcept_no,)).fetchone()
            if (old and old[0] == source["sha256"] and old[1] == "extracted"
                    and (json.loads(old[2]).get("business_passages")
                         or json.loads(old[2]).get("extractor_version", 1) >= 2)):
                evidence = json.loads(old[2])
                result["reused"] += 1
            else:
                try:
                    evidence = read_receipt(path, row.rcept_no, row.corp_code)
                    evidence["extraction_status"] = "extracted"
                    evidence["extractor_version"] = 2
                    error = ""
                    result["new_extractions"] += 1
                except Exception as exc:
                    error = type(exc).__name__
                    evidence = {"extraction_status": "failed", "error_type": error}
                    result["extraction_failed"] += 1
                with state:
                    state.execute("INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?)",
                        (row.rcept_no, source["sha256"], evidence["extraction_status"], json.dumps(evidence, ensure_ascii=False), error))
            rows.append({**base, **evidence})
            if index % 100 == 0:
                log.line(f"processed={index}/{len(initial)}; new_extractions={result['new_extractions']}; reused={result['reused']}; failed={result['extraction_failed']}; industry_decisions=0", timestamp=True)
        output = INTERIM_DATA_DIR / f"industry_evidence_queue_{stamp}.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        result.update({"queue": str(output.relative_to(PROJECT_ROOT)),
            "queue_rows": len(rows), "queue_sha256": file_sha256(output),
            "rows_with_business_passages": sum(bool(r.get("business_passages")) for r in rows),
            "outside_listing_interval": sum(r["within_listing_interval"] is False for r in rows),
            "unknown_listing_interval": sum(r["within_listing_interval"] is None for r in rows),
            "ksic_edition_counts": {str(e): sum(r["ksic_edition"] == e for r in rows) for e in (9, 10, 11)},
            "database_integrity": state.execute("PRAGMA quick_check").fetchone()[0]})
        log.line(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        result["stop_error_type"] = type(exc).__name__
        log.line(f"review_queue_stopped={type(exc).__name__}; checkpoint_preserved=True")
        return 1
    finally:
        state.close()
        archives.close()
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.line(f"report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
