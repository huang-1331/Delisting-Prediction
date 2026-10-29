"""Algorithm 2: portable V15 weight/feature selection, no research data paths.

Fits user-supplied labels; it does not equate delisting with equity impairment.
No pretrained coefficients, data, reports, or previous performance are embedded.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from ...config import ALGORITHM2_WEIGHT_GRID, ALGORITHM_C_GRID, ALGORITHM_SEED

BASE = ['impairment_ratio', 'debt_assets', 'current_ratio']
ADDED = ['roa_ttm','operating_margin_ttm','retained_assets','loss_fraction_4q',
         'equity_delta_4q_assets','leverage_delta_4q','revenue_growth_4q']
LEVEL = ['working_capital_assets','current_assets_assets','current_liabilities_assets',
         'noncurrent_liabilities_assets','noncurrent_assets_assets','retained_equity',
         'equity_assets','asset_turnover_ttm','net_margin_ttm','operating_roa_ttm',
         'operating_income_liabilities','net_income_liabilities','roe_ttm',
         'current_liabilities_total','retained_capital']
TEMPORAL = ['debt_assets_delta_1q','current_ratio_delta_1q','impairment_delta_1q',
            'roa_delta_1q','opmargin_delta_1q','assets_growth_4q','liabilities_growth_4q',
            'current_assets_growth_4q','current_liabilities_growth_4q',
            'net_income_trend_4q_assets','operating_income_trend_4q_assets',
            'wc_assets_delta_4q','retained_assets_delta_4q','current_ratio_delta_4q',
            'opmargin_delta_4q','impairment_mean_4q','current_ratio_mean_4q',
            'roa_mean_4q','opmargin_mean_4q','debt_assets_std_4q','current_ratio_std_4q',
            'roa_std_4q','opmargin_std_4q','longest_loss_streak_4q','negative_operating_fraction_4q']
COLS = BASE+ADDED+LEVEL+TEMPORAL
FOCUSED = ['impairment_ratio','debt_assets','current_ratio','roa_ttm',
           'operating_margin_ttm','retained_assets','loss_fraction_4q',
           'equity_delta_4q_assets','revenue_growth_4q','working_capital_assets',
           'asset_turnover_ttm','impairment_delta_1q','current_ratio_delta_1q',
           'roa_delta_1q','net_income_trend_4q_assets','liabilities_growth_4q',
           'roa_std_4q','negative_operating_fraction_4q']
DEPENDENCIES = {
    'loss_buffer_pressure':['loss_fraction_4q','impairment_ratio'],
    'erosion_buffer_pressure':['impairment_delta_1q','impairment_ratio'],
    'liquidity_loss_interaction':['current_ratio','loss_fraction_4q'],
    'leverage_loss_interaction':['debt_assets','loss_fraction_4q'],
    'profit_erosion_interaction':['roa_ttm','impairment_delta_1q'],
}
FEATURES = {'all50':COLS,'core10':COLS[:10],
    'profitability':BASE+['roa_ttm','operating_margin_ttm','retained_assets',
        'loss_fraction_4q','net_margin_ttm','operating_roa_ttm',
        'operating_income_liabilities','net_income_liabilities','roe_ttm'],
    'focused18':FOCUSED,
    'trend':BASE+COLS[25:40]+['loss_fraction_4q','roa_std_4q','negative_operating_fraction_4q'],
    'focused23_interactions':FOCUSED+list(DEPENDENCIES)}


def required_inputs(columns):
    """Resolve selected feature names into actual required CSV columns."""
    out=set()
    for c in columns:out.update(DEPENDENCIES.get(c,[c]))
    return sorted(out)


def engineer(d, columns):
    """Compute requested five V15 interactions from current/past input ratios."""
    d=d.copy()
    for c in columns:
        if c=='loss_buffer_pressure':d[c]=d.loss_fraction_4q/((.5-d.impairment_ratio).clip(lower=0)+.1)
        elif c=='erosion_buffer_pressure':d[c]=d.impairment_delta_1q.clip(lower=0)/((.5-d.impairment_ratio).clip(lower=0)+.1)
        elif c=='liquidity_loss_interaction':d[c]=(1-d.current_ratio).clip(lower=0)*d.loss_fraction_4q
        elif c=='leverage_loss_interaction':d[c]=d.debt_assets*d.loss_fraction_4q
        elif c=='profit_erosion_interaction':d[c]=-d.roa_ttm*d.impairment_delta_1q.clip(lower=0)
    return d


def weights(d, alpha):
    """Equalize company mass; alpha 0 means no class balance and 1 full balance."""
    w=1/d.groupby('corp_code').corp_code.transform('size').to_numpy(float);total=w.sum()
    for y in [0,1]:
        mask=d.y.to_numpy()==y;mass=w[mask].sum()/total
        if mass<=0:raise ValueError('Both training classes required')
        w[mask]*=(.5/mass)**alpha
    return w*len(d)/w.sum()


def precision_threshold(y,p,min_recall):
    """Maximize validation precision at tied-score boundaries subject to recall."""
    y=np.asarray(y,int);p=np.asarray(p,float)
    if not 0<min_recall<=1 or y.sum()==0 or not np.isfinite(p).all():
        raise ValueError('Invalid recall floor or validation data')
    order=np.argsort(-p,kind='stable');s=p[order];labels=y[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(labels)[ends];alerts=ends+1;recall=tp/y.sum();precision=tp/alerts
    feasible=np.flatnonzero(recall>=min_recall)
    k=max(feasible,key=lambda i:(precision[i],recall[i],-alerts[i]))
    return dict(threshold=float(s[ends[k]]),validation_precision=float(precision[k]),
                validation_recall=float(recall[k]),validation_alerts=int(alerts[k]))


def apply_preprocessing(d, state):
    """Use only the medians and Z-score parameters stored from TRAIN."""
    x=d[state['columns']].replace([np.inf,-np.inf],np.nan)
    return ((x.fillna(pd.Series(state['median']))-pd.Series(state['mean']))/pd.Series(state['scale'])).to_numpy(float)


def fit_system(train, validation, minimum_recall=.5, seed=ALGORITHM_SEED):
    """Search the fixed V15 6x5x3 grid using TRAIN fits and VALIDATION selection.

    Returns a JSON-serializable selected model. No test data enters this API.
    The validation recall floor is not guaranteed on test or future data.
    """
    train=engineer(train,list(DEPENDENCIES));validation=engineer(validation,list(DEPENDENCIES))
    best=None;chosen=None
    for name,cols in FEATURES.items():
        x=train[cols].replace([np.inf,-np.inf],np.nan);med=x.median()
        if med.isna().any():raise ValueError(f'Entire TRAIN feature missing: {list(med[med.isna()].index)}')
        filled=x.fillna(med);mean=filled.mean();sd=filled.std(ddof=0);scale=sd.where(sd.gt(0),1.)
        prep=dict(columns=cols,median=med.to_dict(),mean=mean.to_dict(),scale=scale.to_dict())
        xt=apply_preprocessing(train,prep);xv=apply_preprocessing(validation,prep)
        for alpha in ALGORITHM2_WEIGHT_GRID:
            w=weights(train,alpha)
            for c in ALGORITHM_C_GRID:
                fit=LogisticRegression(C=c,max_iter=3000,solver='lbfgs',random_state=seed).fit(xt,train.y,sample_weight=w)
                p=expit(xv@fit.coef_[0]+fit.intercept_[0]);choice=precision_threshold(validation.y,p,minimum_recall)
                rank=(choice['validation_precision'],choice['validation_recall'],-len(cols),-c,-alpha)
                if best is None or rank>best:
                    best=rank;chosen=dict(algorithm=2,implementation='V15 precision constrained selection',
                        columns=cols,preprocessing=prep,coef=fit.coef_[0].tolist(),intercept=float(fit.intercept_[0]),
                        feature_set=name,alpha=alpha,C=c,minimum_validation_recall=minimum_recall,
                        selection=choice,threshold=choice['threshold'],candidate_count=90,
                        score_semantics='uncalibrated_ranking_score_not_event_probability')
    return chosen


def predict_system(state,d):
    """Score user data using only the selected inputs and saved training state."""
    d=engineer(d,state['columns']);z=apply_preprocessing(d,state['preprocessing'])
    return expit(z@np.asarray(state['coef'])+state['intercept'])
