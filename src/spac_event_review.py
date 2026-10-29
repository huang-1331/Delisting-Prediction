"""Review the 24 overlapping event phrases using receipt-specific public evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from zipfile import ZipFile

from lxml import html
import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import RequestStore, file_sha256
from .config import (PROJECT_ROOT, RAW_DATA_DIR, INTERIM_DATA_DIR,
                     COLLECTION_STATE_PATH, MARKET_PROBE_STATE_PATH,
                     EVENT_OBSERVATION_END)
from .receipt_archive_collection import verify_archive


def purpose_evidence(payload: bytes) -> list[dict]:
    """Find the reporting company's sole-business-purpose statement with an XPath.

    Potential merger targets and their manufacturing industries are not used to
    classify the reporting corporation. Matches are source evidence for review,
    never automatic production labels.
    """
    # Some official ZIPs declare UTF-8 despite containing other source bytes.
    # Only force UTF-8 after strict validation; otherwise retain HTML metadata.
    try:
        payload.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        encoding = None
    tree = html.fromstring(payload, parser=html.HTMLParser(no_network=True, encoding=encoding))
    findings = []
    for node in tree.xpath("//p"):
        text = re.sub(r"\s+", " ", " ".join(node.itertext())).strip()
        compact = re.sub(r"\s+", "", text)
        if ("유일한사업목적" in compact and "합병" in compact
                and any(subject in compact for subject in ("당사는", "회사는", "당사", "본회사"))):
            anchor = re.search(r"유일한\s*사업\s*목적", text)
            start = max(0, anchor.start() - 160) if anchor else 0
            end = min(len(text), anchor.end() + 90) if anchor else min(len(text), 300)
            findings.append({"xpath": tree.getroottree().getpath(node),
                "text": text[start:end], "text_offset": start,
                "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    return findings


def main() -> int:
    """Verify all reviewed T receipts without new network requests or label changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-reviewed-findings", action="store_true",
        help="Record the already reviewed sole-business-purpose findings for this fixed cohort")
    args = parser.parse_args()
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_spac_review_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    financial = RequestStore(COLLECTION_STATE_PATH)
    archives = RequestStore(MARKET_PROBE_STATE_PATH)
    event_path = RAW_DATA_DIR / f"krx_kosdaq_delisted_2015_{EVENT_OBSERVATION_END}.csv"
    result = {"at_kst": now.isoformat(), "network_requests": 0,
        "event_cutoff": EVENT_OBSERVATION_END, "production_label_rule_changed": False,
        "task03_complete": False, "observation_evidence": [], "cases": []}
    try:
        events = pd.read_csv(INTERIM_DATA_DIR / "event_keyword_overlap_review_20261028_210312_935030.csv", dtype="string")
        manifest = pd.read_csv(INTERIM_DATA_DIR / "annual_receipt_versions_20261028_204138_976560.csv", dtype="string")
        linked = pd.read_csv(INTERIM_DATA_DIR / "collection_v2_linked_2026-10-28_8dc723d3e083.csv", dtype="string")
        selected = manifest.loc[manifest.stock_code.isin(events.stock_code)
            & manifest.version_kind.eq("initial_candidate")].copy()
        if len(events) != 24 or selected.stock_code.nunique() != 24:
            raise ValueError("Reviewed cohort or receipt coverage changed")
        if selected.duplicated(["corp_code", "stock_code", "fiscal_year_reported", "fiscal_month_reported"]).any():
            raise ValueError("Multiple plain-title versions need review")
        result["event_source"] = str(event_path.relative_to(PROJECT_ROOT))
        result["event_source_sha256"] = file_sha256(event_path)
        for event in events.sort_values("stock_code").itertuples(index=False):
            identity = linked.loc[linked.stock_code.eq(event.stock_code)]
            if len(identity) != 1 or identity.iloc[0].collection_status != "ready":
                raise ValueError("Event company identity needs review")
            corp = identity.iloc[0].corp_code
            key = f"{corp}|{event.stock_code}|2015-01-01|{EVENT_OBSERVATION_END}|A"
            if financial.status("disclosures", key) != "success":
                raise ValueError("Company disclosure list not fully collected")
            disclosures = financial.frame("disclosures", key)
            observations = selected.loc[selected.stock_code.eq(event.stock_code)].sort_values(["fiscal_year_reported", "rcept_no"])
            case = {"stock_code": event.stock_code, "company_name": event.company_name,
                "delisting_date": event.delisting_date, "official_event_reason": event.delisting_reason,
                "event_type": "spac_merger_application_deadline_failure",
                "completed_merger_verified": False,
                "industry_at_delisting_for_comparison_only": event.industry,
                "reviewed_observations": len(observations), "purpose_evidence_found": 0}
            for row in observations.itertuples(index=False):
                receipt, public_date = row.rcept_no, pd.Timestamp(row.rcept_dt)
                if not (pd.Timestamp(event.listing_date) <= public_date < pd.Timestamp(event.delisting_date)):
                    raise ValueError("Receipt outside listing interval")
                source = disclosures.loc[disclosures.rcept_no.astype(str).eq(receipt)]
                if len(source) != 1 or pd.Timestamp(source.iloc[0].rcept_dt) != public_date:
                    raise ValueError("Receipt public date mismatch")
                if archives.status("dart_receipt_original", receipt) != "success":
                    raise ValueError("Original receipt archive unavailable")
                metadata = archives.frame("dart_receipt_original", receipt).iloc[0]
                path = PROJECT_ROOT / metadata.path
                verify_archive(path, metadata.sha256)
                with ZipFile(path) as archive:
                    main_name = f"{receipt}.xml"
                    if main_name not in archive.namelist():
                        raise ValueError("Main document filename requires review")
                    raw = archive.read(main_name)
                evidence = purpose_evidence(raw)
                additional_sources = []
                if not evidence:
                    fallback = INTERIM_DATA_DIR / f"spac_viewer_source_evidence_{receipt}.json"
                    if fallback.exists():
                        for item in json.loads(fallback.read_text(encoding="utf-8")):
                            if item["receipt"] != receipt:
                                raise ValueError("Fallback source receipt mismatch")
                            section_path = PROJECT_ROOT / item["path"]
                            if file_sha256(section_path) != item["sha256"]:
                                raise ValueError("Fallback source hash mismatch")
                            findings = purpose_evidence(section_path.read_bytes())
                            for finding in findings:
                                finding["source_document"] = item["path"]
                                finding["source_url"] = item["url"]
                                finding["source_sha256"] = item["sha256"]
                            evidence.extend(findings)
                            additional_sources.append({k: item[k] for k in
                                ("receipt", "url", "path", "sha256", "collected_at_utc")})
                observation = {"stock_code": event.stock_code, "corp_code": corp,
                    "fiscal_year": int(row.fiscal_year_reported),
                    "fiscal_month": int(row.fiscal_month_reported),
                    "T": public_date.date().isoformat(), "receipt_no": receipt,
                    "public_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}",
                    "archive": metadata.path, "archive_sha256": metadata.sha256,
                    "document": main_name, "business_purpose_evidence": evidence,
                    "additional_official_sources": additional_sources,
                    "assessment": "review_pending", "manufacturing": None,
                    "financial_values_first_version_verified": False}
                result["observation_evidence"].append(observation)
                case["purpose_evidence_found"] += int(bool(evidence))
            result["cases"].append(case)
            log.line(f"review_stock={event.stock_code}; original_reports={len(observations)}; main_business_purpose_matches={case['purpose_evidence_found']}; new_requests=0")
        result["companies"] = len(result["cases"])
        result["observations"] = len(result["observation_evidence"])
        result["observations_with_business_purpose_evidence"] = sum(bool(x["business_purpose_evidence"]) for x in result["observation_evidence"])
        result["source_crosschecks_passed"] = True
        if args.record_reviewed_findings:
            if (result["companies"], result["observations"], result["observations_with_business_purpose_evidence"]) != (24, 64, 64):
                raise ValueError("Reviewed cohort evidence must be complete before recording findings")
            rows = []
            for observation in result["observation_evidence"]:
                # This flag records the agent's completed source review of this
                # fixed cohort; it is not an automatic industry classifier.
                observation["assessment"] = "reviewed_sole_business_spac_at_T"
                observation["manufacturing"] = 0
                evidence = min(observation["business_purpose_evidence"], key=lambda x: len(x["text"]))
                rows.append({"corp_code": observation["corp_code"], "stock_code": observation["stock_code"],
                    "T": observation["T"], "fiscal_year": observation["fiscal_year"],
                    "receipt_no": observation["receipt_no"], "manufacturing_confirmed": 0,
                    "review_status": "reviewed_original_business_purpose",
                    "reason": "sole_business_is_spac_merger_preparation",
                    "source_url": evidence.get("source_url", observation["public_url"]),
                    "source_path": evidence.get("source_document", observation["archive"]),
                    "source_sha256": evidence.get("source_sha256", observation["archive_sha256"]),
                    "document": evidence.get("source_document", observation["document"]),
                    "xpath": evidence["xpath"],
                    "source_text_sha256": evidence["source_text_sha256"]})
            output = INTERIM_DATA_DIR / f"industry_review_spac_{stamp}.csv"
            pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
            result["reviewed_industry_artifact"] = str(output.relative_to(PROJECT_ROOT))
            result["reviewed_nonmanufacturing_observations"] = len(rows)
            log.line(f"reviewed_nonmanufacturing_observations={len(rows)}; scope=reviewed_T_only; no_global_company_blacklist=True")
        return 0
    except Exception as exc:
        result["review_error_type"] = type(exc).__name__
        log.line(f"source_review_error_type={type(exc).__name__}")
        return 1
    finally:
        financial.close()
        archives.close()
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.line(f"reviewed_companies={len(result['cases'])}; reviewed_receipts={len(result['observation_evidence'])}; label_rules_changed=False; report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
