"""Altman Z-Score baseline implementation.

The project uses Altman Z-Score as the first baseline before developing
machine-learning classifiers. For public manufacturing firms, the common
original form is:

Z = 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 1.0 * X5

where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value of Equity / Total Liabilities
    X5 = Sales / Total Assets

TODO:
    Confirm whether the final study should use the original public
    manufacturing formula, emerging-market variant, or a Korea-specific
    adaptation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ALTMAN_REQUIRED_COLUMNS = [
    "working_capital",
    "retained_earnings",
    "ebit",
    "market_value_equity",
    "sales",
    "total_assets",
    "total_liabilities",
]


def validate_altman_columns(df: pd.DataFrame) -> None:
    """Validate that all columns required for Altman Z-Score exist."""

    missing_columns = [col for col in ALTMAN_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for Altman Z-Score: {missing_columns}")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series and return NaN when the denominator is zero."""

    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def calculate_altman_z_score(df: pd.DataFrame) -> pd.Series:
    """Calculate Altman Z-Score for each row in a financial dataframe.

    Args:
        df: DataFrame containing only financial values observable at time `T`.

    Returns:
        A pandas Series containing the Altman Z-Score for each row.
    """

    validate_altman_columns(df)

    x1 = safe_divide(df["working_capital"], df["total_assets"])
    x2 = safe_divide(df["retained_earnings"], df["total_assets"])
    x3 = safe_divide(df["ebit"], df["total_assets"])
    x4 = safe_divide(df["market_value_equity"], df["total_liabilities"])
    x5 = safe_divide(df["sales"], df["total_assets"])

    return (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)


def add_altman_z_score(
    df: pd.DataFrame,
    output_column: str = "altman_z_score",
) -> pd.DataFrame:
    """Return a copy of `df` with an Altman Z-Score column added."""

    result = df.copy()
    result[output_column] = calculate_altman_z_score(result)
    return result


def classify_altman_zone(z_score: pd.Series) -> pd.Series:
    """Classify Altman Z-Score into distress, grey, and safe zones.

    Common cutoffs for the original public manufacturing formula are:
        - Z < 1.81: distress
        - 1.81 <= Z < 2.99: grey
        - Z >= 2.99: safe

    TODO:
        Confirm final cutoff policy for Korean KOSDAQ manufacturing firms.
    """

    return pd.Series(
        np.select(
            [z_score < 1.81, z_score < 2.99],
            ["distress", "grey"],
            default="safe",
        ),
        index=z_score.index,
        name="altman_zone",
    )

