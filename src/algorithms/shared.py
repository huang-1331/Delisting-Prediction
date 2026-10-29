"""CSV validation and information-date gates; never generate outcome labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

META = ['corp_code', 'anchor', 'available_at', 'known_procedure_date']


def read_csv(path, features, target=None):
    """Load numeric financial inputs with caller-declared availability metadata.

    `available_at` must be the latest actual publication time among all inputs.
    A blank known_procedure_date denotes no proceeding known in caller's source.
    TODO: Validate original filing vintages, universe and outcome definitions in
    the caller's data pipeline; these cannot be inferred from a feature CSV.
    """
    d = pd.read_csv(path, dtype={'corp_code': str})
    required = set(META + list(features) + ([target] if target else []))
    missing = required-set(d.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    if d.corp_code.isna().any() or d.corp_code.str.strip().eq('').any():
        raise ValueError('corp_code cannot be blank')
    for c in ['anchor', 'available_at', 'known_procedure_date']:
        d[c] = pd.to_datetime(d[c], errors='raise', utc=True)
    if d[['anchor', 'available_at']].isna().any().any():
        raise ValueError('anchor and available_at must be actual known dates')
    if d.duplicated(['corp_code', 'anchor']).any():
        raise ValueError('Duplicate company/anchor: resolve original observations')
    for c in features:
        d[c] = pd.to_numeric(d[c], errors='raise')
    if target:
        if not d[target].isin([0, 1]).all():
            raise ValueError('Outcome must be 0/1; unknown/censored is not normal')
        d['y'] = d[target].astype(int)
    return d


def eligibility(d):
    """Mark rows using only dates at the decision anchor, never future events."""
    status = pd.Series('eligible', index=d.index, dtype=object)
    status.loc[d.available_at.gt(d.anchor)] = 'inputs_not_available_at_anchor'
    known = d.known_procedure_date.notna() & d.known_procedure_date.le(d.anchor)
    status.loc[known] = 'procedure_already_known'
    return status.eq('eligible'), status


def disjoint(train, validation, paired=False):
    """Reject company overlap, and pair overlap for the original paired model."""
    if set(train.corp_code) & set(validation.corp_code):
        raise ValueError('TRAIN and VALIDATION must use disjoint companies')
    if paired and set(train.pair_id) & set(validation.pair_id):
        raise ValueError('TRAIN and VALIDATION must use disjoint pair IDs')


def eligible_training(d, paired=False):
    """Screen known procedures/future inputs; paired mode removes entire pairs."""
    keep, _ = eligibility(d)
    if paired:
        if 'pair_id' not in d or d.pair_id.isna().any():
            raise ValueError('Algorithm 1 requires nonempty pair_id')
        counts = d.groupby('pair_id').y.agg(['size', 'sum'])
        if not ((counts['size'] == 2) & (counts['sum'] == 1)).all():
            raise ValueError('Each pair must contain one event and one normal firm')
        rejected = set(d.loc[~keep, 'pair_id'])
        keep &= ~d.pair_id.isin(rejected)
    result = d[keep].copy().reset_index(drop=True)
    if result.y.nunique() != 2:
        raise ValueError('Eligible TRAIN/VALIDATION must contain both classes')
    return result, dict(input_rows=len(d), eligible_rows=len(result), excluded_rows=len(d)-len(result))


def save_json(path, value):
    """Write a JSON model or metric record supplied by this run only."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def result_frame(d, probabilities, threshold, status):
    """Return aligned scores and warnings, with ineligible rows explicitly blank."""
    eligible = status.eq('eligible')
    out = d[['corp_code', 'anchor']].copy()
    if 'company' in d:
        out['company'] = d.company
    out['risk_score'] = np.where(eligible, probabilities, np.nan)
    out['warning'] = np.where(eligible, (np.asarray(probabilities) >= threshold).astype(float), np.nan)
    out['status'] = status.to_numpy()
    return out
