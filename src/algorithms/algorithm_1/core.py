"""PAIR-Guard v1: paired-CV feature experts with a baseline anchor.

Scores describe the matched case-control sample, not market event probabilities.
No test labels enter this module's fitting/selection functions.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

BASE = ['impairment_ratio', 'debt_assets', 'current_ratio']
ADDED = ['roa_ttm', 'operating_margin_ttm', 'retained_assets',
         'loss_fraction_4q', 'equity_delta_4q_assets',
         'leverage_delta_4q', 'revenue_growth_4q']
GROUPS = {'baseline': BASE, **{f'P{i:02d}': BASE + list(p)
          for i, p in enumerate(combinations(ADDED, 2), 1)}, 'all10': BASE + ADDED}
CS = [.1, 1., 10.]


def paired_splits(d, folds=5, seed=20261028):
    """A matched case and control always stay together; each firm appears once."""
    if d.corp_code.duplicated().any():
        raise ValueError('Duplicate company')
    groups = d.groupby('pair_id').y.agg(['size', 'sum'])
    if not ((groups['size'] == 2) & (groups['sum'] == 1)).all():
        raise ValueError('Expected one case and one control per pair')
    pairs = groups.index.to_numpy()
    for a, b in KFold(min(folds, len(pairs)), shuffle=True, random_state=seed).split(pairs):
        yield np.flatnonzero(d.pair_id.isin(pairs[a])), np.flatnonzero(d.pair_id.isin(pairs[b]))


def fit_expert(d, columns, c):
    x = d[columns].to_numpy(float)
    x[~np.isfinite(x)] = np.nan
    if np.any(np.mean(np.isfinite(x), axis=0) < .8):
        raise ValueError('Training feature coverage below 80%')
    median = np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), median, x)
    low, high = np.quantile(x, [.01, .99], axis=0)
    x = np.clip(x, low, high)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.
    m = LogisticRegression(C=c, max_iter=3000, solver='lbfgs').fit((x-mean)/scale, d.y)
    return dict(columns=columns, C=c, median=median.tolist(), low=low.tolist(),
                high=high.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]))


def predict_expert(m, d):
    x = d[m['columns']].to_numpy(float)
    x = np.where(np.isfinite(x), x, m['median'])
    z = (np.clip(x, m['low'], m['high'])-m['mean'])/m['scale']
    return expit(z @ np.asarray(m['coef']) + m['intercept'])


def sweet_threshold(y, p):
    """Prefer recall>=.70 and FPR<=.30, then G-mean; never use test labels."""
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for t in np.unique(np.r_[0., p, .5, 1.]):
        pred = p >= t
        recall = float(pred[y == 1].mean())
        fpr = float(pred[y == 0].mean())
        rows.append(dict(threshold=float(t), recall=recall, fpr=fpr,
                         feasible=bool(recall >= .7 and fpr <= .3),
                         gmean=float(np.sqrt(recall*(1-fpr)))))
    return max(rows, key=lambda r: (r['feasible'], r['gmean'], -r['fpr'],
                                    -abs(r['threshold']-.5), r['threshold']))


def fit_system(d, seed=20261028):
    """Select regularization and experts ONLY on inner paired OOF predictions.

    For each variable group, 1-SE rule chooses strongest eligible regularization.
    At most five near-best groups are averaged; fixed 25% baseline anchor.
    No tuned meta-learner is fitted to its own in-sample predictions.
    """
    splits = list(paired_splits(d, folds=4, seed=seed))
    rows, oof, group_choices = [], {}, {}
    for name, columns in GROUPS.items():
        options = []
        for c in CS:
            p = np.empty(len(d)); losses = []
            for a, b in splits:
                m = fit_expert(d.iloc[a], columns, c)
                p[b] = predict_expert(m, d.iloc[b])
                losses.append(float(log_loss(d.iloc[b].y, p[b], labels=[0, 1])))
            r = dict(group=name, C=c, mean_loss=float(np.mean(losses)),
                     se=float(np.std(losses, ddof=1)/np.sqrt(len(losses))))
            rows.append(r); options.append((r, p))
        best = min(options, key=lambda z: z[0]['mean_loss'])[0]
        selected, p = min((z for z in options if z[0]['mean_loss'] <= best['mean_loss']+best['se']),
                          key=lambda z: z[0]['C'])
        group_choices[name] = selected; oof[name] = p
    ordered = sorted(group_choices, key=lambda n: (group_choices[n]['mean_loss'],
                                                  len(GROUPS[n]), n))
    best = group_choices[ordered[0]]
    shortlist = [n for n in ordered if group_choices[n]['mean_loss'] <= best['mean_loss']+best['se']][:5]
    raw = np.exp(-np.array([group_choices[n]['mean_loss']-best['mean_loss'] for n in shortlist])/.05)
    weights = {n: float(.75*w/raw.sum()) for n, w in zip(shortlist, raw)}
    weights['baseline'] = weights.get('baseline', 0.)+.25
    needed = set(weights) | {'baseline', 'all10'}
    fitted = {n: fit_expert(d, GROUPS[n], group_choices[n]['C']) for n in sorted(needed)}
    state = dict(algorithm='PAIR-Guard-v1', weights=weights, experts=fitted,
                 group_choices=group_choices, shortlist=shortlist,
                 score_semantics='matched_case_control_score_NOT_market_event_probability',
                 verified_for_deployment=False)
    oof['integrated'] = sum(w*oof[n] for n, w in weights.items())
    return state, oof, rows


def predict_system(state, d, variant='integrated'):
    if variant != 'integrated':
        return predict_expert(state['experts'][variant], d)
    return sum(w*predict_expert(state['experts'][n], d) for n, w in state['weights'].items())


def predict_frame(state, d):
    """Conservative missing-data gate for user-facing inference."""
    required = sorted(set(c for n in state['weights'] for c in state['experts'][n]['columns']))
    if not set(required).issubset(d.columns):
        raise ValueError(f'Missing columns: {sorted(set(required)-set(d.columns))}')
    missing = ~np.isfinite(d[required].to_numpy(float))
    eligible = (missing.mean(axis=1) <= .2) & np.isfinite(d[BASE].to_numpy(float)).all(axis=1)
    p = predict_system(state, d)
    result = pd.DataFrame(index=d.index)
    for c in ['corp_code', 'company']:
        if c in d: result[c] = d[c]
    result['risk_score_0_100'] = np.where(eligible, p*100, np.nan)
    result['warning'] = np.where(eligible, (p >= state['threshold']).astype(float), np.nan)
    result['status'] = np.where(eligible, 'exploratory_score', 'insufficient_data')
    return result
