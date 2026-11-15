"""Twelve-month complete impairment, publication-date features and fixed policies."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "모델 검증"))

from cascade import measure, threshold_options
from final_model import dataset
from full_impairment import normalize_panel
from legacy_inputs import SEED

RULES = ("impairment_ratio", "equity_buffer", "annual_loss_projection",
         "quarterly_equity_projection", "annual_equity_projection", "combined_projection")
COLUMNS = ("equity_buffer", "annual_loss_projection", "quarterly_equity_projection",
           "annual_equity_projection", "impairment_ratio", "liquidity",
           "earnings_pressure", "debt_assets", "recovery")


def prepare(path, history, registry):
    eligible, _ = dataset(path, "full", history, registry)
    eligible["anchor"] = pd.to_datetime(eligible.anchor)
    features = []
    groups = {key: g for key,g in history.groupby("corp_code")}
    for row in eligible.itertuples():
        known = groups[row.corp_code]
        known = known.loc[known.available.le(row.anchor)].set_index("quarter").sort_index()
        if known.empty:
            features.append({})
            continue
        q = known.index[-1]
        latest = known.iloc[-1]
        previous = known.reindex([q-1]).iloc[0]
        year_ago = known.reindex([q-4]).iloc[0]
        quarter_flows = known.reindex(range(q-3,q+1)).net_income.copy()
        for k in quarter_flows.index:
            if pd.Timestamp(pd.Period(ordinal=k,freq="Q").end_time).month == 12:
                quarter_flows.loc[k] -= known.reindex(range(k-3,k)).net_income.sum(min_count=3)
        annual = latest.net_income if latest.period_end.month==12 else quarter_flows.sum(min_count=4)
        assets = latest.assets
        equity = latest.equity/assets
        annual_result = annual/assets
        recent_move = (latest.equity-previous.equity)/assets
        annual_move = (latest.equity-year_ago.equity)/assets
        forecasts = pd.Series([equity+annual_result,equity+4*recent_move,equity+annual_move])
        features.append(dict(
            equity_buffer=-equity,
            impairment_ratio=1-latest.equity/latest.capital if latest.capital>0 else np.nan,
            annual_loss_projection=-(equity+annual_result),
            quarterly_equity_projection=-(equity+4*recent_move),
            annual_equity_projection=-(equity+annual_move),
            combined_projection=-forecasts.dropna().median() if forecasts.notna().any() else np.nan,
            liquidity=(latest.current_liabilities-latest.current_assets)/assets,
            earnings_pressure=-annual_result, debt_assets=latest.liabilities/assets,
            recovery=recent_move,
        ))
    return eligible, pd.DataFrame(features).replace([np.inf,-np.inf],np.nan)


def rules(x):
    return {key:x[key].fillna(x.equity_buffer).fillna(-1e6).to_numpy() for key in RULES}


def fit(data, history, registry, out):
    frame,x=prepare(data,history,registry)
    y=frame.y_12m.to_numpy(dtype=int)
    firms=frame.groupby("corp_code").y_12m.max()
    scores=rules(x)
    scores["compact_logistic"]=np.zeros(len(y))
    models=[]
    for rep in range(2):
        folds=StratifiedKFold(3,shuffle=True,random_state=SEED+rep)
        for tr,va in folds.split(firms,firms):
            train=frame.corp_code.isin(firms.index[tr])
            validation=frame.corp_code.isin(firms.index[va])
            pipe=make_pipeline(SimpleImputer(strategy="median",add_indicator=True),
                               RobustScaler(),LogisticRegression(C=.05,max_iter=2000))
            pipe.fit(x.loc[train,list(COLUMNS)].clip(-5,5),y[train])
            scores["compact_logistic"][validation]+=pipe.predict_proba(
                x.loc[validation,list(COLUMNS)].clip(-5,5))[:,1]/2
            models.append(pipe)
    policies=[]
    for name,score in scores.items():
        for floor in (.8,.85,.9):
            cut=threshold_options(y,np.zeros(len(y),bool),np.ones(len(y),bool),
                                  score,int(np.ceil(floor*y.sum())))
            policies.append(dict(model=name,floor=floor,cut=cut,**measure(y,score>=cut)))
    primary=max((p for p in policies if p["floor"]==.8),key=lambda p:(p["precision"],p["recall"]))
    out.mkdir(parents=True,exist_ok=True)
    (out/"lock.json").write_text(json.dumps(dict(primary=primary,policies=policies),indent=2)+"\n")
    joblib.dump(dict(models=models,companies=sorted(firms.index)),out/"models.joblib")
    print("development primary",primary,flush=True)


def test(data,history,registry,out):
    frame,x=prepare(data,history,registry)
    lock=json.loads((out/"lock.json").read_text())
    state=joblib.load(out/"models.joblib")
    if set(state["companies"]) & set(frame.corp_code):
        raise ValueError("Training and test companies overlap")
    scores=rules(x)
    scores["compact_logistic"]=np.mean([m.predict_proba(x[list(COLUMNS)].clip(-5,5))[:,1]
                                          for m in state["models"]],axis=0)
    y=frame.y_12m.to_numpy(dtype=int)
    rows=[]
    for row in lock["policies"]:
        score=scores[row["model"]]
        alert=score>=row["cut"]
        rows.append(dict(model=row["model"],floor=row["floor"],cut=row["cut"],**measure(y,alert)))
        frame[f'{row["model"]}_{row["floor"]}_score']=score
        frame[f'{row["model"]}_{row["floor"]}_alert']=alert
    table=pd.DataFrame(rows)
    table.to_csv(out/"test.csv",index=False)
    frame.to_csv(out/"predictions.csv.gz",index=False)
    print(table.to_string(index=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("fit","test"))
    for arg in ("data","history","registry","out"):
        p.add_argument(arg,type=Path)
    a=p.parse_args()
    (fit if a.phase=="fit" else test)(a.data,normalize_panel(a.history),a.registry,a.out)
