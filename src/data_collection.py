"""Point-in-time data collection utilities for the delisting study.

The module collects only official KIND, KRX, and OpenDART data. Raw records
retain source identifiers and collection timestamps so that revised filings
can be audited later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from http.cookiejar import CookieJar
from io import BytesIO, StringIO
import json
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import (
    HTTPCookieProcessor,
    Request,
    build_opener,
    urlopen,
)
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd

try:
    from .config import (
        ANNUAL_REPORT_CODE,
        get_dart_api_key,
        get_krx_credentials,
    )
except ImportError:  # pragma: no cover - supports direct notebook imports
    from config import ANNUAL_REPORT_CODE, get_dart_api_key, get_krx_credentials


DART_API_BASE_URL = "https://opendart.fss.or.kr/api"
KIND_KOSDAQ_LIST_URL = (
    "https://kind.krx.co.kr/corpgeneral/corpList.do"
    "?method=download&searchType=13&marketType=kosdaqMkt"
)
KRX_LOGIN_PAGE = (
    "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd"
)
KRX_LOGIN_FRAME = (
    "https://data.krx.co.kr/contents/MDC/COMS/client/view/login.jsp?site=mdc"
)
KRX_LOGIN_URL = (
    "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001D1.cmd"
)
KRX_DATA_URL = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
KRX_DELISTING_BLD = "dbms/MDC/STAT/issue/MDCSTAT23801"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

STANDARD_LISTING_COLUMNS = [
    "stock_code",
    "company_name",
    "market",
    "industry",
    "listing_date",
    "delisting_date",
    "delisting_reason",
    "security_type",
    "stock_type",
    "source",
    "collected_at_utc",
]


class DartApiError(RuntimeError):
    """Raised when OpenDART returns an error status."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(f"OpenDART error {status}: {message}")
        self.status = status
        self.message = message


class KrxAuthenticationError(RuntimeError):
    """Raised when a KRX authenticated session cannot be established."""


@dataclass(frozen=True)
class DartConnectionResult:
    """Result object returned by OpenDART connection checks."""

    ok: bool
    message: str
    sample: Any | None = None


@dataclass(frozen=True)
class FinancialCollectionResult:
    """Collected annual statements and recoverable collection issues."""

    statements: pd.DataFrame
    issues: pd.DataFrame


@dataclass(frozen=True)
class UniverseLinkResult:
    """Listing history linked to DART codes plus unresolved records."""

    linked: pd.DataFrame
    unresolved: pd.DataFrame


def _utc_timestamp() -> str:
    """Return a reproducible ISO timestamp for collection metadata."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_stock_codes(series: pd.Series) -> pd.Series:
    """Normalize KRX short codes without altering alphanumeric codes."""

    values = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    numeric = values.str.fullmatch(r"\d+") & values.str.len().lt(6)
    values.loc[numeric] = values.loc[numeric].str.zfill(6)
    return values


def _normalize_company_names(series: pd.Series) -> pd.Series:
    """Normalize legal-name decorations for exact, non-fuzzy matching."""

    return (
        series.astype("string")
        .str.replace("주식회사", "", regex=False)
        .str.replace("(주)", "", regex=False)
        .str.replace(r"[\s·.,]", "", regex=True)
        .str.strip()
    )


def _require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    """Raise a clear error when a dataframe contract is incomplete."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")


def _read_url_bytes(
    url: str,
    *,
    opener: Any | None = None,
    data: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> bytes:
    """Read bytes from an HTTP endpoint with an injectable opener."""

    encoded = urlencode(data).encode("utf-8") if data is not None else None
    request_headers = {"User-Agent": USER_AGENT}
    request_headers.update(headers or {})
    request = Request(url, data=encoded, headers=request_headers)
    handler = opener.open if opener is not None else urlopen
    response = handler(request, timeout=timeout)
    with response:
        return response.read()


def create_dart_client(api_key: str | None = None) -> "DartApiClient":
    """Create the project's direct OpenDART REST client.

    Args:
        api_key: Optional API key. If omitted, the local secrets file is used.

    Returns:
        An initialized client that does not make a request until called.
    """

    return DartApiClient(api_key=api_key or get_dart_api_key(required=True))


class DartApiClient:
    """Small OpenDART client for corporation and annual statement data."""

    def __init__(
        self,
        api_key: str,
        *,
        opener: Any | None = None,
        timeout: int = 30,
    ) -> None:
        """Initialize the client with a secret key and injectable HTTP opener."""

        if not api_key:
            raise ValueError("api_key must not be empty")
        self._api_key = api_key
        self._opener = opener
        self._timeout = timeout

    def _get_json(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        query = urlencode({"crtfc_key": self._api_key, **params})
        payload = _read_url_bytes(
            f"{DART_API_BASE_URL}/{endpoint}?{query}",
            opener=self._opener,
            timeout=self._timeout,
        )
        result = json.loads(payload.decode("utf-8"))
        status = str(result.get("status", ""))
        if status != "000":
            raise DartApiError(status, str(result.get("message", "Unknown error")))
        return result

    def fetch_corp_codes(self) -> pd.DataFrame:
        """Download and parse the official OpenDART corporation code archive."""

        query = urlencode({"crtfc_key": self._api_key})
        payload = _read_url_bytes(
            f"{DART_API_BASE_URL}/corpCode.xml?{query}",
            opener=self._opener,
            timeout=self._timeout,
        )
        with ZipFile(BytesIO(payload)) as archive:
            xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
            if len(xml_names) != 1:
                raise ValueError("OpenDART corporation archive must contain one XML file")
            root = ElementTree.fromstring(archive.read(xml_names[0]))

        rows = [
            {child.tag: (child.text or "").strip() for child in item}
            for item in root.findall("list")
        ]
        frame = pd.DataFrame(rows).rename(
            columns={
                "corp_code": "corp_code",
                "corp_name": "company_name",
                "corp_eng_name": "company_name_en",
                "stock_code": "stock_code",
                "modify_date": "modify_date",
            }
        )
        if not frame.empty:
            frame["stock_code"] = _normalize_stock_codes(frame["stock_code"])
            frame["modify_date"] = pd.to_datetime(
                frame["modify_date"], format="%Y%m%d", errors="coerce"
            )
        frame["source"] = "OpenDART corpCode.xml"
        frame["collected_at_utc"] = _utc_timestamp()
        return frame

    def fetch_document_archive(self, receipt_no: str) -> bytes:
        """Return the official original document ZIP for an explicit receipt ID."""
        if not str(receipt_no).isdigit() or len(str(receipt_no)) != 14:
            raise ValueError("A 14-digit receipt number is required")
        query = urlencode({"crtfc_key": self._api_key, "rcept_no": str(receipt_no)})
        payload = _read_url_bytes(f"{DART_API_BASE_URL}/document.xml?{query}",
                                  opener=self._opener, timeout=self._timeout)
        if not payload.startswith(b"PK"):
            raise ValueError("DART did not return a document archive")
        return payload

    def fetch_disclosures(
        self,
        corp_code: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Collect all disclosure-list pages for one corporation and period."""

        rows: list[dict[str, Any]] = []
        page_no = 1
        while True:
            result = self._get_json(
                "list.json",
                {
                    "corp_code": str(corp_code),
                    "bgn_de": start_date.strftime("%Y%m%d"),
                    "end_de": end_date.strftime("%Y%m%d"),
                    "pblntf_ty": "A",
                    "page_no": str(page_no),
                    "page_count": "100",
                },
            )
            rows.extend(result.get("list", []))
            if page_no >= int(result.get("total_page", 1)):
                break
            page_no += 1

        frame = pd.DataFrame(rows)
        if "rcept_dt" in frame.columns:
            frame["rcept_dt"] = pd.to_datetime(
                frame["rcept_dt"], format="%Y%m%d", errors="raise"
            )
        frame["source"] = "OpenDART list.json"
        frame["collected_at_utc"] = _utc_timestamp()
        return frame

    def fetch_financial_statement(
        self,
        corp_code: str,
        business_year: int,
        *,
        report_code: str = ANNUAL_REPORT_CODE,
        fs_div: str = "CFS",
    ) -> pd.DataFrame:
        """Collect all accounts for one annual filing and statement scope."""

        if fs_div not in {"CFS", "OFS"}:
            raise ValueError("fs_div must be 'CFS' or 'OFS'")
        result = self._get_json(
            "fnlttSinglAcntAll.json",
            {
                "corp_code": str(corp_code),
                "bsns_year": str(business_year),
                "reprt_code": report_code,
                "fs_div": fs_div,
            },
        )
        frame = pd.DataFrame(result.get("list", []))
        if not frame.empty:
            frame["fs_div_used"] = fs_div
            frame["business_year_requested"] = business_year
            frame["report_code_requested"] = report_code
        frame["source"] = "OpenDART fnlttSinglAcntAll.json"
        frame["collected_at_utc"] = _utc_timestamp()
        return frame


def fetch_annual_statement_with_fallback(
    client: DartApiClient,
    corp_code: str,
    business_year: int,
) -> pd.DataFrame:
    """Collect an annual statement using CFS first and OFS only when absent."""

    for fs_div in ("CFS", "OFS"):
        try:
            frame = client.fetch_financial_statement(
                corp_code,
                business_year,
                report_code=ANNUAL_REPORT_CODE,
                fs_div=fs_div,
            )
        except DartApiError as exc:
            if exc.status == "013":
                continue
            raise
        if not frame.empty:
            return frame
    return pd.DataFrame()


def collect_annual_financial_statements(
    client: DartApiClient,
    companies: pd.DataFrame,
    business_years: Iterable[int],
) -> FinancialCollectionResult:
    """Collect 2015-2024-style annual statements for an explicit company set.

    Args:
        client: Authenticated OpenDART client.
        companies: Dataframe containing ``corp_code`` and ``stock_code``.
        business_years: Explicit years to collect; callers control study scope.

    Returns:
        Statement rows and a separate issue log. Missing filings are recorded
        rather than silently converted into observations.
    """

    _require_columns(companies, ["corp_code", "stock_code"], "DART collection")
    frames: list[pd.DataFrame] = []
    issues: list[dict[str, Any]] = []
    for company in companies[["corp_code", "stock_code"]].drop_duplicates().itertuples(
        index=False
    ):
        for business_year in business_years:
            try:
                frame = fetch_annual_statement_with_fallback(
                    client, str(company.corp_code), int(business_year)
                )
            except Exception as exc:  # caller receives an auditable issue row
                issues.append(
                    {
                        "corp_code": company.corp_code,
                        "stock_code": company.stock_code,
                        "business_year": business_year,
                        "issue": str(exc),
                    }
                )
                continue
            if frame.empty:
                issues.append(
                    {
                        "corp_code": company.corp_code,
                        "stock_code": company.stock_code,
                        "business_year": business_year,
                        "issue": "No CFS or OFS annual statement returned",
                    }
                )
                continue
            frame["stock_code"] = company.stock_code
            frames.append(frame)

    statements = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return FinancialCollectionResult(statements, pd.DataFrame(issues))


def test_dart_connection(company_name: str = "삼성전자") -> DartConnectionResult:
    """Test OpenDART by downloading the corporation code list."""

    try:
        client = create_dart_client()
        codes = client.fetch_corp_codes()
        sample = codes.loc[codes["company_name"].eq(company_name)].head(1)
    except Exception as exc:  # pragma: no cover - external service
        return DartConnectionResult(False, f"OpenDART connection failed: {exc}")
    return DartConnectionResult(True, "OpenDART connection succeeded.", sample)


def fetch_kind_current_kosdaq_listings(*, opener: Any | None = None) -> pd.DataFrame:
    """Download the current KOSDAQ company list from official KIND."""

    payload = _read_url_bytes(KIND_KOSDAQ_LIST_URL, opener=opener)
    html = payload.decode("euc-kr")
    tables = pd.read_html(
        StringIO(html),
        converters={"종목코드": lambda value: str(value).strip()},
    )
    if len(tables) != 1:
        raise ValueError("KIND listing download must contain exactly one table")
    frame = tables[0].rename(
        columns={
            "회사명": "company_name",
            "시장구분": "market",
            "종목코드": "stock_code",
            "업종": "industry",
            "상장일": "listing_date",
        }
    )
    _require_columns(
        frame,
        ["company_name", "market", "stock_code", "industry", "listing_date"],
        "KIND listings",
    )
    frame["stock_code"] = _normalize_stock_codes(frame["stock_code"])
    frame["listing_date"] = pd.to_datetime(frame["listing_date"], errors="raise")
    frame["delisting_date"] = pd.NaT
    frame["delisting_reason"] = pd.NA
    frame["security_type"] = "주권"
    frame["stock_type"] = "보통주"
    frame["source"] = "KIND current listed companies"
    frame["collected_at_utc"] = _utc_timestamp()
    return frame[STANDARD_LISTING_COLUMNS]


class KrxDataClient:
    """Authenticated KRX Data Marketplace client for delisting history."""

    def __init__(
        self,
        login_id: str | None = None,
        password: str | None = None,
        *,
        opener: Any | None = None,
        timeout: int = 30,
    ) -> None:
        """Initialize credentials without logging or persisting secrets."""

        if login_id is None or password is None:
            login_id, password = get_krx_credentials(required=True)
        self._login_id = str(login_id)
        self._password = str(password)
        self._opener = opener or build_opener(HTTPCookieProcessor(CookieJar()))
        self._timeout = timeout
        self._authenticated = False

    def _request_json(
        self,
        url: str,
        *,
        data: dict[str, str] | None = None,
        referer: str = KRX_LOGIN_PAGE,
    ) -> dict[str, Any]:
        payload = _read_url_bytes(
            url,
            opener=self._opener,
            data=data,
            headers={"Referer": referer, "X-Requested-With": "XMLHttpRequest"},
            timeout=self._timeout,
        )
        text = payload.decode("utf-8", errors="replace").strip()
        if text == "LOGOUT":
            raise KrxAuthenticationError("KRX session is not authenticated")
        return json.loads(text)

    def authenticate(self) -> None:
        """Create a KRX session, including duplicate-login confirmation."""

        _read_url_bytes(KRX_LOGIN_PAGE, opener=self._opener, timeout=self._timeout)
        _read_url_bytes(
            KRX_LOGIN_FRAME,
            opener=self._opener,
            headers={"Referer": KRX_LOGIN_PAGE},
            timeout=self._timeout,
        )
        payload = {
            "mbrNm": "",
            "telNo": "",
            "di": "",
            "certType": "",
            "mbrId": self._login_id,
            "pw": self._password,
        }
        result = self._request_json(KRX_LOGIN_URL, data=payload)
        if result.get("_error_code") == "CD011":
            result = self._request_json(
                KRX_LOGIN_URL, data={**payload, "skipDup": "Y"}
            )
        if result.get("_error_code") != "CD001":
            raise KrxAuthenticationError(
                f"KRX login failed: {result.get('_error_message', 'unknown error')}"
            )
        self._authenticated = True

    def fetch_delisted_issues(
        self,
        start_date: date,
        end_date: date,
        *,
        market: str = "KSQ",
    ) -> pd.DataFrame:
        """Collect official KRX delisting records for a date interval."""

        if start_date > end_date:
            raise ValueError("start_date must not be after end_date")
        if market not in {"ALL", "STK", "KSQ", "KNX"}:
            raise ValueError("market must be one of ALL, STK, KSQ, or KNX")
        if not self._authenticated:
            self.authenticate()
        result = self._request_json(
            KRX_DATA_URL,
            data={
                "bld": KRX_DELISTING_BLD,
                "locale": "ko_KR",
                "mktId": market,
                "isuCd": "ALL",
                "isuCd2": "ALL",
                "strtDd": start_date.strftime("%Y%m%d"),
                "endDd": end_date.strftime("%Y%m%d"),
                "share": "1",
                "money": "1",
            },
            referer="https://data.krx.co.kr/contents/MDC/STAT/issue/MDCSTAT238.jsp",
        )
        frame = pd.DataFrame(result.get("output", []))
        if frame.empty:
            return pd.DataFrame(columns=STANDARD_LISTING_COLUMNS)
        frame = frame.rename(
            columns={
                "ISU_CD": "stock_code",
                "ISU_NM": "company_name",
                "MKT_NM": "market",
                "IDX_IND_NM": "industry",
                "LIST_DD": "listing_date",
                "DELIST_DD": "delisting_date",
                "DELIST_RSN_DSC": "delisting_reason",
                "SECUGRP_NM": "security_type",
                "KIND_STKCERT_TP_NM": "stock_type",
            }
        )
        _require_columns(
            frame,
            [
                "stock_code",
                "company_name",
                "market",
                "industry",
                "listing_date",
                "delisting_date",
                "delisting_reason",
            ],
            "KRX delistings",
        )
        frame["stock_code"] = _normalize_stock_codes(frame["stock_code"])
        frame["listing_date"] = pd.to_datetime(
            frame["listing_date"], format="%Y/%m/%d", errors="coerce"
        )
        frame["delisting_date"] = pd.to_datetime(
            frame["delisting_date"], format="%Y/%m/%d", errors="raise"
        )
        frame["source"] = "KRX Data Marketplace screen 20037"
        frame["collected_at_utc"] = _utc_timestamp()
        return frame[STANDARD_LISTING_COLUMNS]

    def fetch_daily_market(self, trade_date: date) -> pd.DataFrame:
        """Fetch an official KOSDAQ daily snapshot with native prices and shares.

        The MDCSTAT01501 contract was observed in the official MDCSTAT015.jsp
        screen. Returned rows retain source keys, requested date and timestamp;
        they do not establish that any firm's complete equity value is available.
        """
        if not self._authenticated:
            self.authenticate()
        result = self._request_json(KRX_DATA_URL, data={
            "bld": "dbms/MDC/STAT/standard/MDCSTAT01501", "locale": "ko_KR",
            "mktId": "KSQ", "trdDd": trade_date.strftime("%Y%m%d"),
            "share": "1", "money": "1"},
            referer="https://data.krx.co.kr/contents/MDC/STAT/standard/MDCSTAT015.jsp")
        tables = [value for value in result.values() if isinstance(value, list)]
        required = {"ISU_SRT_CD", "TDD_CLSPRC", "LIST_SHRS", "MKTCAP", "ACC_TRDVOL"}
        valid = [pd.DataFrame(rows) for rows in tables if rows and required.issubset(rows[0])]
        if len(valid) == 1:
            frame = valid[0]
        elif len(tables) == 1 and not tables[0]:
            return pd.DataFrame()
        else:
            raise ValueError("KRX daily market contract not satisfied")
        frame["requested_trade_date"] = trade_date.isoformat()
        frame["source"] = "KRX MDCSTAT01501"
        frame["collected_at_utc"] = _utc_timestamp()
        return frame

    def fetch_price_history(self, isin: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch an ISIN's official daily history; retain raw source values.

        Uses the observed MDCSTAT017 contract. Only trading dates and volume
        are used for the freshness investigation until price-adjustment policy
        is independently verified.
        """
        if not self._authenticated:
            self.authenticate()
        if start_date > end_date or not isin:
            raise ValueError("Invalid historical request")
        result = self._request_json(KRX_DATA_URL, data={
            "bld": "dbms/MDC/STAT/standard/MDCSTAT01701", "locale": "ko_KR",
            "isuCd": isin, "strtDd": start_date.strftime("%Y%m%d"),
            "endDd": end_date.strftime("%Y%m%d"), "share": "1", "money": "1"},
            referer="https://data.krx.co.kr/contents/MDC/STAT/standard/MDCSTAT017.jsp")
        tables = [value for value in result.values() if isinstance(value, list)]
        required = {"TRD_DD", "ACC_TRDVOL", "TDD_CLSPRC"}
        valid = [pd.DataFrame(rows) for rows in tables if rows and required.issubset(rows[0])]
        if len(valid) == 1:
            frame = valid[0]
        elif len(tables) == 1 and not tables[0]:
            return pd.DataFrame()
        else:
            raise ValueError("KRX history contract not satisfied")
        frame["source"] = "KRX MDCSTAT01701"
        frame["collected_at_utc"] = _utc_timestamp()
        frame["requested_isin"] = isin
        frame["requested_start_date"] = start_date.isoformat()
        frame["requested_end_date"] = end_date.isoformat()
        frame["price_adjustment_status"] = "source_default_unverified"
        return frame


def build_kosdaq_manufacturing_history(
    current_listings: pd.DataFrame,
    delisted_issues: pd.DataFrame,
) -> pd.DataFrame:
    """Return identity-eligible collection candidates, not a final manufacturing sample.

    Compatibility entry point: industry at T remains unknown until evidence is
    reviewed. Use build_collection_candidates to retain excluded/review records.
    """
    candidates = build_collection_candidates(current_listings, delisted_issues)
    return candidates.loc[candidates["collection_status"].eq("ready")].reset_index(drop=True)


def build_collection_candidates(
    current_listings: pd.DataFrame,
    delisted_issues: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve official rows and classify collection eligibility independently of industry.

    Inputs are current and historical official listing tables. Output includes
    ready, identity_review and excluded rows. Ready means KOSDAQ common equity
    is established; it never means manufacturing at T has been established.
    """
    for name, frame in (("current", current_listings), ("delisted", delisted_issues)):
        _require_columns(frame, STANDARD_LISTING_COLUMNS, name)
    result = pd.concat([current_listings, delisted_issues], ignore_index=True).copy()
    result["stock_code"] = _normalize_stock_codes(result["stock_code"])
    for column in ("listing_date", "delisting_date"):
        result[column] = pd.to_datetime(result[column], errors="raise")
    # Keep distinct conflicting records for review instead of choosing a winner.
    result = result.drop_duplicates(subset=[c for c in STANDARD_LISTING_COLUMNS if c != "collected_at_utc"])
    market = result["market"].astype("string").str.strip()
    security = result["security_type"].astype("string").str.strip()
    stock_type = result["stock_type"].astype("string").str.strip()
    kosdaq = market.str.fullmatch("코스닥|KOSDAQ", case=False, na=False)
    known = market.notna() & market.ne("") & security.notna() & security.ne("") & stock_type.notna() & stock_type.ne("")
    ready = kosdaq & security.eq("주권").fillna(False) & stock_type.eq("보통주").fillna(False)
    result["collection_status"] = "identity_review"
    excluded = ((market.notna() & market.ne("") & ~kosdaq)
                | (security.notna() & security.ne("") & security.ne("주권"))
                | (stock_type.notna() & stock_type.ne("") & stock_type.ne("보통주"))).fillna(False)
    result.loc[excluded, "collection_status"] = "excluded"
    result.loc[ready & known, "collection_status"] = "ready"
    invalid = (~result["stock_code"].str.fullmatch(r"[0-9A-Za-z]{6}", na=False)
               | result["listing_date"].isna()
               | (result["delisting_date"].notna() & result["delisting_date"].le(result["listing_date"])))
    result.loc[invalid & ~excluded, "collection_status"] = "identity_review"
    keys = ["stock_code", "listing_date", "delisting_date"]
    conflict = result.groupby(keys, dropna=False)[["market", "security_type", "stock_type"]].transform("nunique").gt(1).any(axis=1)
    result.loc[conflict, "collection_status"] = "identity_review"
    result["manufacturing_confirmed"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result["industry_verification_status"] = "pending_at_T"
    result["final_sample_eligible"] = False
    # TODO: consume reviewed firm-T evidence only after its schema is finalized.
    return result.sort_values(keys, na_position="last").reset_index(drop=True)


def filter_observations_to_listing_window(
    observations: pd.DataFrame,
    listing_history: pd.DataFrame,
    *,
    stock_code_column: str = "stock_code",
    observation_date_column: str = "observation_date",
) -> pd.DataFrame:
    """Keep only observations made while the corresponding issue was listed."""

    _require_columns(
        observations,
        [stock_code_column, observation_date_column],
        "point-in-time observations",
    )
    _require_columns(
        listing_history,
        ["stock_code", "listing_date", "delisting_date"],
        "listing history",
    )
    obs = observations.copy()
    obs[stock_code_column] = _normalize_stock_codes(obs[stock_code_column])
    obs[observation_date_column] = pd.to_datetime(
        obs[observation_date_column], errors="raise"
    )
    history = listing_history[["stock_code", "listing_date", "delisting_date"]].copy()
    history["stock_code"] = _normalize_stock_codes(history["stock_code"])
    merged = obs.merge(
        history,
        left_on=stock_code_column,
        right_on="stock_code",
        how="left",
        validate="many_to_many",
        suffixes=("", "_history"),
    )
    active = merged[observation_date_column].ge(merged["listing_date"]) & (
        merged["delisting_date"].isna()
        | merged[observation_date_column].lt(merged["delisting_date"])
    )
    return merged.loc[active, observations.columns].drop_duplicates().reset_index(drop=True)


def link_listing_history_to_dart_codes(
    listing_history: pd.DataFrame,
    dart_corp_codes: pd.DataFrame,
) -> UniverseLinkResult:
    """Link KRX/KIND history to DART identifiers without fuzzy matching.

    Stock-code matches take priority. Rows still unmatched may use a normalized
    company-name match only when that name maps to exactly one DART corporation.
    Ambiguous and unmatched rows are returned separately for manual review.
    """

    _require_columns(
        listing_history,
        ["stock_code", "company_name"],
        "listing-to-DART linkage",
    )
    _require_columns(
        dart_corp_codes,
        ["corp_code", "stock_code", "company_name"],
        "DART corporation codes",
    )
    history = listing_history.copy()
    history["_row_id"] = range(len(history))
    history["stock_code"] = _normalize_stock_codes(history["stock_code"])
    history["_normalized_name"] = _normalize_company_names(history["company_name"])

    dart = dart_corp_codes[["corp_code", "stock_code", "company_name"]].copy()
    dart["stock_code"] = _normalize_stock_codes(dart["stock_code"])
    dart["_normalized_name"] = _normalize_company_names(dart["company_name"])

    stock_map = (
        dart.loc[dart["stock_code"].notna() & dart["stock_code"].ne("")]
        .drop_duplicates("stock_code", keep=False)
        .set_index("stock_code")["corp_code"]
    )
    history["corp_code"] = history["stock_code"].map(stock_map).astype("string")
    history["corp_code_match_method"] = pd.Series(
        pd.NA, index=history.index, dtype="string"
    )
    history.loc[history["corp_code"].notna(), "corp_code_match_method"] = "stock_code"

    unique_name_map = (
        dart.loc[dart["_normalized_name"].notna() & dart["_normalized_name"].ne("")]
        .drop_duplicates("_normalized_name", keep=False)
        .set_index("_normalized_name")["corp_code"]
    )
    missing = history["corp_code"].isna()
    ambiguous_codes = set(dart.loc[dart["stock_code"].ne("") & dart["stock_code"].notna()
        & dart.duplicated("stock_code", keep=False), "stock_code"])
    # An ambiguous code must not be silently resolved by falling back to a name.
    name_allowed = missing & ~history["stock_code"].isin(ambiguous_codes)
    history.loc[name_allowed, "corp_code"] = history.loc[
        name_allowed, "_normalized_name"
    ].map(unique_name_map)
    name_matched = missing & history["corp_code"].notna()
    history.loc[name_matched, "corp_code_match_method"] = "company_name_exact"

    unresolved = history.loc[history["corp_code"].isna()].copy()
    linked = history.loc[history["corp_code"].notna()].copy()
    drop_columns = ["_row_id", "_normalized_name"]
    return UniverseLinkResult(
        linked=linked.drop(columns=drop_columns).reset_index(drop=True),
        unresolved=unresolved.drop(columns=drop_columns).reset_index(drop=True),
    )


def attach_disclosure_receipt_dates(
    statements: pd.DataFrame,
    disclosures: pd.DataFrame,
) -> pd.DataFrame:
    """Attach disclosure receipt dates to statement rows as observation T.

    The join uses DART receipt numbers, not fiscal year-end dates. Statement
    rows without a matching receipt date are rejected because their actual
    information-availability time cannot be established.
    """

    _require_columns(statements, ["rcept_no"], "financial statements")
    _require_columns(disclosures, ["rcept_no", "rcept_dt"], "disclosures")
    dates = disclosures[["rcept_no", "rcept_dt"]].drop_duplicates("rcept_no")
    dates["rcept_dt"] = pd.to_datetime(dates["rcept_dt"], errors="raise")
    result = statements.merge(dates, on="rcept_no", how="left", validate="many_to_one")
    if result["rcept_dt"].isna().any():
        missing = sorted(result.loc[result["rcept_dt"].isna(), "rcept_no"].unique())
        raise ValueError(f"Missing disclosure receipt dates for: {missing}")
    return result.rename(columns={"rcept_dt": "observation_date"})
