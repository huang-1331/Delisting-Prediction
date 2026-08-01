"""Leakage-aware target labeling for T+12 month delisting prediction."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

try:
    from .config import TARGET_COLUMN
except ImportError:  # pragma: no cover - supports direct notebook imports
    from config import TARGET_COLUMN


NON_DISTRESS_REASON_KEYWORDS = (
    "합병",
    "주식교환",
    "주식이전",
    "이전상장",
    "자진상장폐지",
    "기업인수목적회사",
    "스팩합병",
)

DISTRESS_REASON_KEYWORDS = (
    "감사의견",
    "의견거절",
    "부적정",
    "자본잠식",
    "부도",
    "파산",
    "회생절차",
    "영업정지",
    "매출액",
    "시가총액",
    "정기보고서 미제출",
    "사업보고서 미제출",
    "공시의무",
    "횡령",
    "배임",
    "상장적격성",
    "기업의 계속성",
    "경영의 투명성",
    "관리종목",
)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    """Validate dataframe columns used by the labeling contract."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")


def classify_delisting_reasons(reasons: pd.Series) -> pd.Series:
    """Classify official delisting reasons as distress, non-distress, or unknown.

    Non-distress events are checked first so merger or transfer events are not
    treated as financial failures. Unrecognized descriptions remain ``NA`` and
    must be reviewed before the final dataset is frozen.

    Args:
        reasons: Official KRX delisting-reason descriptions.

    Returns:
        Nullable boolean values: True for distress, False for an explicit
        competing event, and NA for unresolved reasons.
    """

    normalized = reasons.astype("string").str.replace(r"\s+", "", regex=True)
    result = pd.Series(pd.NA, index=reasons.index, dtype="boolean")
    non_distress = normalized.apply(
        lambda value: False
        if pd.isna(value)
        else any(keyword.replace(" ", "") in value for keyword in NON_DISTRESS_REASON_KEYWORDS)
    )
    distress = normalized.apply(
        lambda value: False
        if pd.isna(value)
        else any(keyword.replace(" ", "") in value for keyword in DISTRESS_REASON_KEYWORDS)
    )
    result.loc[non_distress] = False
    result.loc[~non_distress & distress] = True
    return result.rename("is_distress_delisting")


def list_unresolved_delisting_reasons(
    delistings: pd.DataFrame,
    *,
    reason_column: str = "delisting_reason",
) -> pd.DataFrame:
    """Return unresolved reason descriptions and frequencies for manual review."""

    _require_columns(delistings, [reason_column], "reason audit")
    classified = classify_delisting_reasons(delistings[reason_column])
    unresolved = delistings.loc[classified.isna(), reason_column]
    return (
        unresolved.value_counts(dropna=False)
        .rename_axis(reason_column)
        .reset_index(name="count")
    )


def create_t12_delisting_label(
    observations: pd.DataFrame,
    delistings: pd.DataFrame,
    *,
    label_as_of: str | date | pd.Timestamp,
    company_id_column: str = "stock_code",
    observation_date_column: str = "observation_date",
    delisting_date_column: str = "delisting_date",
    distress_column: str = "is_distress_delisting",
) -> pd.DataFrame:
    """Create a right-censoring-aware 12-month distress-delisting target.

    A distress event strictly after T and on or before T+12 months is labeled
    1. A complete event-free window is labeled 0. Incomplete windows,
    non-distress competing delistings, and unresolved reasons are labeled NA.
    Event dates and reasons are never copied into the model-ready output.

    Args:
        observations: Firm-time rows observable at T.
        delistings: Official event rows with a nullable distress classification.
        label_as_of: Last date through which events have been observed.
        company_id_column: Stable identifier shared by both inputs.
        observation_date_column: Column containing T.
        delisting_date_column: Event date column in ``delistings``.
        distress_column: Nullable boolean event classification.

    Returns:
        A copy of ``observations`` with nullable integer target values.
    """

    _require_columns(
        observations,
        [company_id_column, observation_date_column],
        "T+12 observations",
    )
    _require_columns(
        delistings,
        [company_id_column, delisting_date_column, distress_column],
        "T+12 delistings",
    )
    result = observations.copy()
    result[observation_date_column] = pd.to_datetime(
        result[observation_date_column], errors="raise"
    )
    events = delistings[
        [company_id_column, delisting_date_column, distress_column]
    ].copy()
    events[delisting_date_column] = pd.to_datetime(
        events[delisting_date_column], errors="raise"
    )
    events[distress_column] = events[distress_column].astype("boolean")
    as_of = pd.Timestamp(label_as_of)

    grouped_events = {
        company_id: group.sort_values(delisting_date_column)
        for company_id, group in events.groupby(company_id_column, dropna=False)
    }
    labels: list[object] = []
    for row in result[[company_id_column, observation_date_column]].itertuples(
        index=False, name=None
    ):
        company_id, observation_date = row
        horizon_end = observation_date + pd.DateOffset(months=12)
        if horizon_end > as_of:
            labels.append(pd.NA)
            continue
        company_events = grouped_events.get(company_id)
        if company_events is None:
            labels.append(0)
            continue
        within_window = company_events.loc[
            company_events[delisting_date_column].gt(observation_date)
            & company_events[delisting_date_column].le(horizon_end)
        ]
        if within_window.empty:
            labels.append(0)
            continue
        first_event = within_window.iloc[0][distress_column]
        if pd.isna(first_event):
            labels.append(pd.NA)
        elif bool(first_event):
            labels.append(1)
        else:
            labels.append(pd.NA)

    result[TARGET_COLUMN] = pd.array(labels, dtype="Int8")
    return result
