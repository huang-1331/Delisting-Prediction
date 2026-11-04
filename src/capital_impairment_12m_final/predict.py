"""Score 12-month capital impairment with the frozen v75 research model."""
import argparse
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_SHA256 = "227d8ed28ad72e48784c6dd15f107a7ea1b32017c042dbf3448f7b6cdda221f1"
LABEL_LIMIT = .35
EXTRA = {
    'erosion_pressure_at_label_limit', 'projected_impairment',
    'liability_to_capital', 'loss_to_capital_proxy',
}


def predict(frame):
    """Return the frozen risk score and warning for each point-in-time feature row."""
    path = Path(__file__).with_name('model.joblib')
    if sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError('Frozen model SHA-256 mismatch')
    bundle = joblib.load(path)
    prep = bundle['prep']
    required = (set(prep['columns']) - EXTRA) | {'equity_assets'}
    missing = required - set(frame)
    if missing:
        raise ValueError('Missing input columns: ' + ', '.join(sorted(missing)))

    features = frame[sorted(required)].apply(pd.to_numeric, errors='raise')
    impairment = features.impairment_ratio
    erosion = features.impairment_delta_1q.clip(lower=0)
    features['erosion_pressure_at_label_limit'] = erosion / ((LABEL_LIMIT - impairment).clip(lower=0) + .1)
    features['projected_impairment'] = impairment + 2 * erosion.fillna(0)
    equity_assets = features.equity_assets.where(features.equity_assets.abs() > .01)
    assets_to_capital = ((1 - impairment) / equity_assets).where(lambda value: value > 0)
    features['liability_to_capital'] = features.debt_assets * assets_to_capital
    features['loss_to_capital_proxy'] = -features.roa_ttm * assets_to_capital

    x = features[prep['columns']].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(pd.Series(prep['median']))
    x = ((x - pd.Series(prep['mean'])) / pd.Series(prep['scale'])).to_numpy(float)
    score = bundle['model'].predict_proba(x)[:, 1]
    return pd.DataFrame({'risk_score': score, 'warning': score >= bundle['threshold']}, index=frame.index)


def main():
    """Read an engineered CSV and write a CSV containing scores and warnings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv')
    parser.add_argument('output_csv')
    args = parser.parse_args()
    frame = pd.read_csv(args.input_csv, dtype={'corp_code': str}, float_precision='round_trip')
    keys = [key for key in ('corp_code', 'feature_end', 'anchor') if key in frame]
    pd.concat([frame[keys], predict(frame)], axis=1).to_csv(args.output_csv, index=False)


if __name__ == '__main__':
    main()
