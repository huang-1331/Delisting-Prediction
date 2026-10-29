"""Record a fixed set of individually inspected industry and event-source facts.

This is an auditable record of source reviews, not a keyword industry classifier
or a T+12 target generator. Uninspected cases keep their existing unknown state.
"""

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import PurePosixPath
import re
import sqlite3
from zipfile import ZipFile

from lxml import html
import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import file_sha256
from .config import PROJECT_ROOT, INTERIM_DATA_DIR, RAW_DATA_DIR, MARKET_PROBE_STATE_PATH
from .industry_evidence import ksic_edition_at_T


# Manually inspected receipts. No inference to another T or another company.
PAPER_RECEIPTS = {"20160330001400", "20170331004063", "20180402003753",
                  "20190401004363", "20200330003428"}
SPAC_RECEIPTS = {"20190329000115", "20200325000219", "20210308000292"}
# Individually read parent-company paragraphs. The 2020 S-Mark report discloses
# production cessation and is deliberately outside this reviewed set.
TEXTILE_REVIEW_PATHS = {
    "20160330004289": "/html/body/document/section-1[2]/section-2[1]/p[2]",
    "20170331004229": "/html/body/document/section-1[2]/section-2[1]/p[5]",
    "20180402004048": "/html/body/document/section-1[2]/section-2[1]/p[2]",
    "20190401004704": "/html/body/document/section-1[2]/section-2[1]/p[3]",
}
RETAIL_RECEIPTS = {"20160330001989", "20170330001791", "20180402003077",
                   "20190401000873", "20200330002214", "20210317000112"}
EXCHANGE_REVIEWS = {
    "033630": ("20150609000299", "2015-06-09"),
    "066270": ("20170207000165", "2017-02-07"),
    "225650": ("20190828000510", "2019-08-27"),
    "057500": ("20201208000342", "2020-12-08"),
    "046140": ("20231204000307", "2023-12-04"),
    "085370": ("20231024000203", "2023-10-13"),
    "298870": ("20230808000109", "2023-08-08"),
    "138580": ("20241108000266", "2024-11-08"),
    "287410": ("20241023000083", "2024-10-23"),
    "119860": ("20240909000188", "2024-09-06"),
    "115960": ("20240214001666", "2024-02-14"),
    "029960": ("20250613000412", "2025-05-30"),
    "230360": ("20260615000154", "2026-06-15"),
}
VOLUNTARY_REVIEWS = {
    "053740": ("20150115900189", "2015-01-15"),
    "046110": ("20220831900483", "2022-08-31"),
    "220630": ("20220330901084", "2022-03-30"),
    "048260": ("20230628900267", "2023-06-28"),
    "006580": ("20240517900193", "2024-05-17"),
    "335890": ("20251106900405", "2025-11-06"),
}


def source_tree(path, receipt):
    """Read exactly the reviewed source member with strict character decoding."""
    with ZipFile(path) as archive:
        members = [n for n in archive.namelist() if PurePosixPath(n).name == receipt + ".xml"]
        if len(members) != 1:
            raise ValueError("Reviewed source member not unique")
        raw = archive.read(members[0])
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp949")
    tree = html.fromstring(re.sub(r"^\s*<\?xml[^>]*\?>", "", text))
    return tree, members[0]


def combine_industry_reviews(baseline: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    """Combine only explicitly reviewed binary facts, rejecting duplicate T keys."""
    combined = pd.concat([baseline, additions], ignore_index=True)
    if (combined.duplicated(["corp_code", "stock_code", "T"]).any()
            or not combined.manufacturing_confirmed.astype(str).isin(["0", "1"]).all()):
        raise ValueError("Duplicate, conflicting or unreviewed industry facts")
    return combined


def explicit_full_year_dates(text: str) -> set[str]:
    """Read explicit four-digit-year source dates without guessing missing years."""
    found = set()
    pattern = r"(?<!\d)(\d{4})\s*(?:년|[.\-/])\s*(\d{1,2})\s*(?:월|[.\-/])\s*(\d{1,2})(?:\s*일)?(?!\d)"
    for year, month, day in re.findall(pattern, text):
        found.add(date(int(year), int(month), int(day)).isoformat())
    return found


def additional_parent_reviews(queue: list[dict]) -> pd.DataFrame:
    """Verify ten inspected parent-business assertions without extending to other T.

    Some queue passages describe a nested subsidiary. The textile review uses
    the individually inspected parent paragraph in the unchanged original XML.
    """
    reviewed = []
    for row in queue:
        receipt = row["receipt"]
        if receipt not in set(TEXTILE_REVIEW_PATHS) | RETAIL_RECEIPTS:
            continue
        textile = receipt in TEXTILE_REVIEW_PATHS
        expected = ("030270", "00100258") if textile else ("028150", "00207755")
        if (row["stock_code"], row["corp_code"]) != expected or row["within_listing_interval"] is not True:
            raise ValueError("Additional reviewed identity or listing interval changed")
        path = PROJECT_ROOT / row["archive"]
        if file_sha256(path) != row["archive_sha256"]:
            raise ValueError("Additional reviewed original changed")
        tree, member = source_tree(path, receipt)
        identity = tree.xpath("//document-header/company-name[@aregcik] | //document/company-name[@aregcik]")
        if len(identity) != 1 or identity[0].get("aregcik") != row["corp_code"]:
            raise ValueError("Additional reviewed reporting entity changed")
        if textile:
            nodes = tree.xpath(TEXTILE_REVIEW_PATHS[receipt])
            reason = "Parent major-business paragraph explicitly identifies textile-yarn manufacturing; subsidiary entertainment business is separate"
        else:
            nodes = [n for n in tree.xpath("//p")
                     if "당사의사업부문은TV쇼핑,모바일쇼핑,인터넷쇼핑" in re.sub(r"\s+", "", " ".join(n.itertext()))]
            reason = "Reporting company explicitly identifies its own principal business as online retail and mail-order shopping"
        if len(nodes) != 1:
            raise ValueError("Additional inspected parent paragraph not unique")
        node = nodes[0]
        text = re.sub(r"\s+", " ", " ".join(node.itertext())).strip()
        compact = re.sub(r"\s+", "", text)
        if textile and not ("주요사업의내용" in compact and "원사를전문으로제조하는기업입니다" in compact
                            and ("당사는창사이래" in compact or "지배회사는창사이래" in compact)):
            raise ValueError("Additional inspected parent manufacturing assertion missing")
        reviewed.append({"corp_code": row["corp_code"], "stock_code": row["stock_code"],
            "T": row["T"], "fiscal_year": row["fiscal_year"], "receipt_no": receipt,
            "manufacturing_confirmed": int(textile), "ksic_edition": ksic_edition_at_T(row["T"]),
            "ksic_section": "C" if textile else "not_C", "ksic_division": "13" if textile else "",
            "review_status": "reviewed_original_business_statement", "reason": reason,
            "source_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}",
            "source_path": row["archive"], "source_sha256": row["archive_sha256"],
            "document": member, "xpath": tree.getroottree().getpath(node),
            "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    if len(reviewed) != 10:
        raise ValueError("Additional fixed review set incomplete")
    return pd.DataFrame(reviewed)


def main() -> int:
    """Validate source hashes/identities and write reviewed facts, leaving labels untouched."""
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_verified_reviews_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    result = {"at_kst": now.isoformat(), "network_requests": 0,
        "production_classifier_changed": False, "T12_targets_created": 0,
        "task03_complete": False, "task04_executed": False}
    try:
        queue_path = INTERIM_DATA_DIR / "industry_evidence_queue_20261029_104859_207776.jsonl"
        queue = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
        reviewed_industry = []
        for row in queue:
            receipt = row["receipt"]
            if receipt not in PAPER_RECEIPTS | SPAC_RECEIPTS:
                continue
            result["current_review_receipt"] = receipt
            result["review_phase"] = "industry"
            if not row["within_listing_interval"]:
                raise ValueError("Reviewed T outside listing interval")
            if file_sha256(PROJECT_ROOT / row["archive"]) != row["archive_sha256"]:
                raise ValueError("Reviewed industry source changed")
            expected_stock = "006580" if receipt in PAPER_RECEIPTS else "310840"
            if row["stock_code"] != expected_stock:
                raise ValueError("Reviewed industry identity changed")
            if receipt in PAPER_RECEIPTS:
                phrase = "당사는골판지상자의원재료인골판지원지를제조,판매하고있으며"
                flag, section, division = 1, "C", "17"
                reason = "Company's own major-business paragraph explicitly identifies corrugating-paper manufacture and sale"
            else:
                phrase = "당사는한국거래소의코스닥시장에상장한후다른회사와합병하는것을유일한사업목적으로합니다"
                flag, section, division = 0, "not_C", ""
                reason = "Reporting SPAC's sole business is merger preparation; not its potential target's industry"
            matches = [p for p in row["business_passages"] if phrase in re.sub(r"\s+", "", p["text"])]
            if not matches:
                raise ValueError("Inspected company-business assertion missing")
            evidence = matches[0]
            reviewed_industry.append({"corp_code": row["corp_code"], "stock_code": row["stock_code"],
                "T": row["T"], "fiscal_year": row["fiscal_year"], "receipt_no": receipt,
                "manufacturing_confirmed": flag, "ksic_edition": ksic_edition_at_T(row["T"]),
                "ksic_section": section, "ksic_division": division,
                "review_status": "reviewed_original_business_statement", "reason": reason,
                "source_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}",
                "source_path": row["archive"], "source_sha256": row["archive_sha256"],
                "document": row["member"], "xpath": evidence["xpath"],
                "source_text_sha256": evidence["text_sha256"]})
        if len(reviewed_industry) != 8:
            raise ValueError("Fixed industry review set incomplete")
        new_industry = pd.DataFrame(reviewed_industry)
        baseline_path = INTERIM_DATA_DIR / "industry_review_spac_20261028_215832_963667.csv"
        baseline = pd.read_csv(baseline_path, dtype="string")
        baseline["ksic_edition"] = baseline["T"].map(ksic_edition_at_T)
        combined = combine_industry_reviews(baseline, new_industry)
        additional = additional_parent_reviews(queue)
        combined = combine_industry_reviews(combined, additional)
        industry_output = INTERIM_DATA_DIR / f"industry_review_combined_{stamp}.csv"
        combined.to_csv(industry_output, index=False, encoding="utf-8-sig")

        reason_path = INTERIM_DATA_DIR / "event_reason_review_20261029_103743_040497.csv"
        reasons = pd.read_csv(reason_path, dtype="string")
        manifest = pd.read_csv(INTERIM_DATA_DIR / "event_completion_source_manifest_20261029_105017_907496.csv", dtype="string")
        linked = pd.read_csv(INTERIM_DATA_DIR / "collection_v2_linked_2026-10-28_8dc723d3e083.csv", dtype="string").set_index("stock_code")
        archives = sqlite3.connect(f"file:{MARKET_PROBE_STATE_PATH.as_posix()}?mode=ro", uri=True)
        event_evidence = []
        try:
            for category, cases in [("completed_share_exchange", EXCHANGE_REVIEWS), ("voluntary_delisting_application", VOLUNTARY_REVIEWS)]:
                for stock, (receipt, reviewed_date) in cases.items():
                    result["current_review_receipt"] = receipt
                    result["review_phase"] = category
                    filing = manifest.loc[manifest.rcept_no.eq(receipt)]
                    event = reasons.loc[reasons.stock_code.eq(stock)]
                    if len(filing) != 1 or len(event) != 1 or filing.iloc[0].event_stock_code != stock:
                        raise ValueError("Reviewed event source binding ambiguous")
                    filing, event = filing.iloc[0], event.iloc[0]
                    if filing.corp_code != linked.loc[stock].corp_code:
                        raise ValueError("Reviewed event corporation mismatch")
                    saved = archives.execute("SELECT payload FROM requests WHERE stage='dart_receipt_original' AND request_key=? AND status='success'", (receipt,)).fetchone()
                    if not saved:
                        raise ValueError("Reviewed event archive unavailable")
                    metadata = json.loads(saved[0])["data"][0]
                    path = PROJECT_ROOT / metadata["path"]
                    if file_sha256(path) != metadata["sha256"]:
                        raise ValueError("Reviewed event source bytes changed")
                    tree, member = source_tree(path, receipt)
                    if category == "completed_share_exchange":
                        header = tree.xpath("//company-name[@aregcik]")
                        if len(header) != 1 or header[0].get("aregcik") != filing.corp_code:
                            raise ValueError("Completion report entity mismatch")
                        if "포괄적교환" not in filing.report_nm:
                            raise ValueError("Wrong completion report type")
                        candidates = [n for n in tree.xpath("//tr")
                            if re.match(r"^주식교환일", re.sub(r"\s+", "", " ".join(n.itertext())))]
                        matches = [n for n in candidates if ("23.08.08" in " ".join(n.itertext()) if stock == "298870"
                            else reviewed_date in explicit_full_year_dates(" ".join(n.itertext())))]
                    else:
                        if pd.Timestamp(filing.rcept_dt).date().isoformat() != reviewed_date:
                            raise ValueError("Application date/source mismatch")
                        matches = [n for n in tree.xpath("//td|//p")
                            if "자진상장폐지" in re.sub(r"\s+", "", " ".join(n.itertext()))]
                        if "보통주" not in " ".join(tree.itertext()):
                            raise ValueError("Ordinary-share event not confirmed")
                    if not matches:
                        raise ValueError("Reviewed explicit event evidence missing")
                    evidence = min(matches, key=lambda n: len(" ".join(n.itertext())))
                    text = re.sub(r"\s+", " ", " ".join(evidence.itertext())).strip()
                    event_evidence.append({"stock_code": stock, "delisting_date": event.delisting_date,
                        "reviewed_category": category, "reviewed_distress_event": False,
                        "source_receipt": receipt, "source_public_date": filing.rcept_dt,
                        "reviewed_action_date": reviewed_date,
                        "source_path": metadata["path"], "source_sha256": metadata["sha256"],
                        "document": member, "xpath": tree.getroottree().getpath(evidence),
                        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "T12_target": None})
                    idx = reasons.index[reasons.stock_code.eq(stock)][0]
                    reasons.loc[idx, "reviewed_category"] = category
                    reasons.loc[idx, "reviewed_distress_event"] = "False"
                    reasons.loc[idx, "review_status"] = "reviewed_explicit_official_context"
                    reasons.loc[idx, "context_source_receipt"] = receipt
        finally:
            archives.close()
        # A final corrected KRX notice explicitly distinguishes the postponed
        # dispersion review from the approved voluntary delisting in this case.
        path = RAW_DATA_DIR / "receipt_original_20240610900602.zip"
        result["review_phase"] = "final_voluntary_notice"
        with sqlite3.connect(f"file:{MARKET_PROBE_STATE_PATH.as_posix()}?mode=ro", uri=True) as check:
            source_metadata = check.execute("SELECT payload FROM requests WHERE stage='dart_receipt_original' AND request_key='20240610900602' AND status='success'").fetchone()
        if not source_metadata or file_sha256(path) != json.loads(source_metadata[0])["data"][0]["sha256"]:
            raise ValueError("Final notice source provenance changed")
        tree, member = source_tree(path, "20240610900602")
        body = re.sub(r"\s+", "", " ".join(tree.itertext()))
        for required in ("자진상장폐지를승인", "2024년06월21일", "심의를속개"):
            if required not in body:
                raise ValueError("Final DaeYang notice needs review")
        result["daeyang_final_notice"] = {"receipt": "20240610900602",
            "source_path": str(path.relative_to(PROJECT_ROOT)), "sha256": file_sha256(path),
            "correction_used_for_event_context_only": True,
            "pending_dispersion_proceeding_recorded": True}
        details = INTERIM_DATA_DIR / f"event_context_review_evidence_{stamp}.json"
        details.write_text(json.dumps(event_evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reason_output = INTERIM_DATA_DIR / f"event_reason_context_review_{stamp}.csv"
        reasons.to_csv(reason_output, index=False, encoding="utf-8-sig")
        result.update({"new_reviewed_industry_observations": 18, "new_manufacturing_observations": 9,
            "new_nonmanufacturing_observations": 9, "combined_industry_observations": len(combined),
            "additional_parent_review_observations": len(additional),
            "production_cessation_observations_not_auto_classified": ["20200330001636"],
            "industry_artifact": str(industry_output.relative_to(PROJECT_ROOT)),
            "industry_sha256": file_sha256(industry_output),
            "new_reviewed_competing_events": len(event_evidence),
            "reviewed_distress_events_in_original_unknowns": 10,
            "remaining_unknown_events": int(reasons.review_status.eq("unresolved_official_context_required").sum()),
            "reason_artifact": str(reason_output.relative_to(PROJECT_ROOT)),
            "reason_sha256": file_sha256(reason_output),
            "event_evidence_artifact": str(details.relative_to(PROJECT_ROOT)),
            "unresolved_spac_event_is_not_a_normal_target": True})
        log.line(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        result["stop_error_type"] = type(exc).__name__
        safe_messages = {"Inspected company-business assertion missing", "Reviewed T outside listing interval",
            "Reviewed industry identity changed", "Fixed industry review set incomplete",
            "Reviewed event source binding ambiguous", "Reviewed event corporation mismatch",
            "Reviewed event archive unavailable", "Completion report entity mismatch",
            "Wrong completion report type", "Application date/source mismatch",
            "Ordinary-share event not confirmed", "Reviewed explicit event evidence missing",
            "Final DaeYang notice needs review", "Duplicate, conflicting or unreviewed industry facts"}
        if exc.args and exc.args[0] in safe_messages:
            result["validation_reason"] = exc.args[0]
        log.line(f"fixed_review_stopped={type(exc).__name__}; no_model_inputs_changed=True")
        return 1
    finally:
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.line(f"report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
