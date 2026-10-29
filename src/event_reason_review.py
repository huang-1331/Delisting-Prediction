"""Stage explicit official event-reason variants under already approved categories."""

from datetime import datetime, timedelta, timezone
import json

import pandas as pd

from .collection_runner import ProgressLog
from .collection_state import file_sha256
from .config import PROJECT_ROOT, RAW_DATA_DIR, INTERIM_DATA_DIR, EVENT_OBSERVATION_END
from .labeling import classify_delisting_reasons


# Fixed, inspected source phrases; this is not a generic substring classifier.
# Corporate-control changes or delisting applications alone remain unresolved.
REVIEWED_REASON_CATEGORIES = {
    "자본전액잠식": "capital_impairment",
    "자본전액잠식 등": "capital_impairment",
    "감사범위제한으로 인한 한정의견": "audit_opinion",
    "감사범위 제한": "audit_scope_limitation",
    "최근 5사업연도 연속 영업손실 발생": "financial_loss",
    "최근 2년간 3회 이상 공시규정 제19조제1항의 규정에 의한 사업보고서, 반기보고서 또는 분기보고서 법정제출기한 내 미제출": "periodic_report_non_submission",
}


def reviewed_reason_category(reason: str) -> str | None:
    """Return only an inspected explicit category; ambiguous phrases stay unknown."""
    return REVIEWED_REASON_CATEGORIES.get(reason)


def main() -> int:
    """Preserve source-backed review findings without creating any T+12 target."""
    now = datetime.now(timezone(timedelta(hours=9)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    report = PROJECT_ROOT / "reports/monthly" / f"task03_event_reason_review_{stamp}.json"
    log = ProgressLog(report.with_suffix(".txt"))
    result = {"at_kst": now.isoformat(), "external_requests": 0,
        "production_label_classifier_changed": False, "T12_targets_created": 0,
        "task03_complete": False, "task04_executed": False}
    try:
        source = RAW_DATA_DIR / f"krx_kosdaq_delisted_2015_{EVENT_OBSERVATION_END}.csv"
        digest = file_sha256(source)
        events = pd.read_csv(source, dtype="string")
        common = events.loc[events.security_type.eq("주권") & events.stock_type.eq("보통주")]
        unknown = common.loc[classify_delisting_reasons(common.delisting_reason).isna()].copy()
        unknown["reviewed_category"] = unknown.delisting_reason.map(reviewed_reason_category)
        unknown["review_status"] = unknown.reviewed_category.map(
            lambda category: "reviewed_explicit_approved_category" if pd.notna(category) else "unresolved_official_context_required")
        unknown["reviewed_distress_event"] = pd.Series(
            pd.NA, index=unknown.index, dtype="boolean")
        unknown.loc[unknown.reviewed_category.notna(), "reviewed_distress_event"] = True
        unknown["reason_source_path"] = str(source.relative_to(PROJECT_ROOT))
        unknown["reason_source_sha256"] = digest
        unknown["reviewed_at_kst"] = now.isoformat()
        output = INTERIM_DATA_DIR / f"event_reason_review_{stamp}.csv"
        unknown.to_csv(output, index=False, encoding="utf-8-sig")
        assert file_sha256(source) == digest
        result.update({"source": str(source.relative_to(PROJECT_ROOT)), "source_sha256": digest,
            "review_artifact": str(output.relative_to(PROJECT_ROOT)),
            "review_sha256": file_sha256(output), "initial_unknown_events": len(unknown),
            "explicit_approved_category_events": int(unknown.reviewed_category.notna().sum()),
            "remaining_unknown_events": int(unknown.reviewed_category.isna().sum()),
            "reviewed_category_counts": unknown.reviewed_category.dropna().value_counts().to_dict(),
            "source_preserved": True,
            "notes": "Reviewed event type is separate from industry eligibility and from the T+12 target. Existing 24 overlapping SPAC reasons are a separate review cohort."})
        log.line(json.dumps(result, ensure_ascii=False))
        return 0
    finally:
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.line(f"report={report.relative_to(PROJECT_ROOT)}")
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
