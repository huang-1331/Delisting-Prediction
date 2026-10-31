"""Frozen Algorithm4 inference on 50 precomputed financial features; no fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit,softmax
from scipy.stats import rankdata


def engineer(d):
    d=d.copy();buffer=(.5-d.impairment_ratio).clip(lower=0)+.1
    d['loss_buffer_pressure']=d.loss_fraction_4q/buffer
    d['erosion_buffer_pressure']=d.impairment_delta_1q.clip(lower=0)/buffer
    d['liquidity_loss_interaction']=(1-d.current_ratio).clip(lower=0)*d.loss_fraction_4q
    d['leverage_loss_interaction']=d.debt_assets*d.loss_fraction_4q
    d['profit_erosion_interaction']=-d.roa_ttm*d.impairment_delta_1q.clip(lower=0)
    return d


def probability(model,x):
    """Apply saved binary or multinomial logistic coefficients."""
    z=x@np.asarray(model['coef']).T+np.asarray(model['intercept'])
    if len(model['classes'])==2:
        p=expit(z[:,0]);return np.column_stack([1-p,p])
    return softmax(z,axis=1)


def predict_bundle(bundle,x,mix):
    """Return primary risk, auxiliary state scores and expert disagreement."""
    ix=bundle['feature_indices'];base=probability(bundle['global_model'],x[:,ix])[:,1]
    p=base.copy()
    if bundle['k']>1:
        centers=np.asarray(bundle['centers']);z=x[:,bundle['profile_indices']]
        profiles=((z[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
    else:profiles=np.zeros(len(x),int)
    for name,local in bundle['local_models'].items():
        mask=profiles==int(name)
        if mask.any():p[mask]=(1-local['weight'])*base[mask]+local['weight']*probability(local['model'],x[mask][:,ix])[:,1]
    disagreement=np.abs(base-p)
    if bundle['auxiliary_model'] is not None:
        states=probability(bundle['auxiliary_model'],x[:,ix]);auxrisk=states[:,1]+states[:,2]
        p=(1-mix)*p+mix*auxrisk
        ambiguity=4*states[:,1]*states[:,2]/np.maximum(auxrisk**2,1e-15)*p if mix>0 else np.zeros(len(x))
    else:states=np.full((len(x),3),np.nan);ambiguity=np.zeros(len(x))
    return dict(p=p,states=states,profiles=profiles,disagreement=disagreement,ambiguity=ambiguity)


def transform(frame,prep):
    """Use saved TRAIN-only statistics, never the new batch's statistics."""
    x=engineer(frame)[prep['columns']].replace([np.inf,-np.inf],np.nan)
    return ((x.fillna(pd.Series(prep['median']))-pd.Series(prep['mean']))/pd.Series(prep['scale'])).to_numpy(float)


def predict_ensemble(frame,policy,folder):
    """Apply an exported research candidate, with explicit batch rank semantics."""
    prep=json.loads((Path(folder)/policy['preprocessing_file']).read_text())
    x=transform(frame,prep);scores=[]
    for component in policy['components']:
        if component['kind']=='profile':
            model=json.loads((Path(folder)/component['file']).read_text())
            p=predict_bundle(model['bundle'],x,model['mix'])['p']
        else:
            from xgboost import XGBClassifier
            model=XGBClassifier();model.load_model(Path(folder)/component['file']);p=model.predict_proba(x)[:,1]
        if policy['score_mode']=='rank':p=rankdata(p,method='average')/len(p)
        scores.append(component['weight']*p)
    score=np.sum(scores,axis=0)
    return score,score>=policy['threshold']


def predict(frame, horizon=12):
    """Return frozen research scores and flags for a frame of 50 raw features."""
    if horizon not in (12,24):raise ValueError('horizon must be 12 or 24')
    if frame.empty:raise ValueError('input must contain at least one row')
    folder=Path(__file__).resolve().parent/'models'
    policy=json.loads((folder/f'ensemble_{horizon}_rank1.json').read_text())
    prep=json.loads((folder/policy['preprocessing_file']).read_text())
    required=[c for c in prep['columns'] if c not in ['loss_buffer_pressure','erosion_buffer_pressure','liquidity_loss_interaction','leverage_loss_interaction','profit_erosion_interaction']]
    missing=sorted(set(required)-set(frame.columns))
    if missing:raise ValueError('Missing feature columns: '+', '.join(missing))
    score,flag=predict_ensemble(frame,policy,folder)
    result=frame[[c for c in ['corp_code','company','feature_end','anchor'] if c in frame]].copy()
    result['horizon_months']=horizon
    result['research_score']=score
    result['warning']=flag
    result['threshold']=policy['threshold']
    return result
