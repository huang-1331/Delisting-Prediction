"""Feature engineering utilities.

All generated features must be based only on information observable at time
`T`. Any feature requiring future financial statements, post-event market data,
or delisting outcomes must be excluded.
"""

from __future__ import annotations

import pandas as pd


def build_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Build financial ratio features from point-in-time statements.

    TODO:
        Implement after the OpenDART extraction schema is finalized. Expected
        ratios include liquidity, leverage, profitability, activity, and growth
        indicators that can be calculated at time `T`.
    """

    return df.copy()

