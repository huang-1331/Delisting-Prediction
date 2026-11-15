"""Company-disjoint 12-month distress model selection and held-out evaluation."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cascade import measure, threshold_options
from full_impairment import normalize_panel
from legacy_inputs import matrices, SEED
from revise import financial_additions

SPEC = {
    "logistic": ("logistic", False, 0),
    "logistic_expanded": ("logistic", True, 0),
    "trees": ("trees", False, 0),
    "trees_expanded": ("trees", True, 0),
    "boosting": ("boosting", False, 0),
    "boosting_expanded": ("boosting", True, 0),
    "trees_hard_negative": ("trees", True, 3),
    "boosting_hard_negative": ("boosting", True, 3),
}
BLENDS = {
    "logistic_trees": ("logistic_expanded", "trees_expanded"),
    "logistic_boosting": ("logistic_expanded", "boosting_expanded"),
    "three_models": ("logistic_expanded", "trees_expanded", "boosting_expanded"),
    "two_tree_models": ("trees_hard_negative", "boosting_expanded"),
}


def published_inputs(frame, history):
    keys = ["corp_code", "anchor"]
    columns = ["assets", "equity", "capital", "liabilities", "current_assets", "current_liabilities", "period_end"]
    requests = frame[keys].copy()
    requests["anchor"] = pd.to_datetime(requests.anchor)
    requests["row"] = requests.index
    dates = requests.sort_values("anchor")
    filings = history.sort_values(["available", "period_end"]).copy()
    # A late older filing must not replace a newer fiscal quarter already known.
    latest_period = filings.groupby("corp_code").period_end.cummax()
    filings = filings[filings.period_end.eq(latest_period)]
    known = pd.merge_asof(dates, filings[["corp_code", "available", *columns]], left_on="anchor",
                          right_on="available", by="corp_code", direction="backward")
    known = known.set_index("row").reindex(frame.index)
    assert not known.available.gt(known.anchor).any()
    equity = known.equity / known.assets
    capital = known.capital.where(known.capital.gt(0))
    ratio = 1 - known.equity / capital
    return pd.DataFrame({
        "latest_equity_assets": equity,
        "latest_capital_assets": capital/known.assets,
        "latest_impairment_ratio": ratio,
        "latest_liabilities_assets": known.liabilities/known.assets,
        "latest_current_liabilities_assets": known.current_liabilities/known.assets,
        "latest_current_ratio": known.current_assets/known.current_liabilities.where(known.current_liabilities.gt(0)),
        "latest_working_capital_assets": (known.current_assets-known.current_liabilities)/known.assets,
        "latest_equity_change_assets": equity-(1-frame.debt_assets),
        "latest_impairment_change": ratio-frame.impairment_ratio,
        "latest_fiscal_age_days": (known.anchor-known.period_end).dt.days,
    }, index=frame.index)


def dataset(path, mode, history, registry):
    frame = pd.read_csv(path, dtype={"corp_code":str})
    if frame.duplicated(["corp_code","anchor"]).any() or frame.feature_end.gt(frame.anchor).any():
        raise ValueError("Invalid snapshot keys")
    base = matrices(frame)["trajectory"]
    expanded = base.join(financial_additions(frame,base))
    eligible = frame.y_12m.notna()
    if mode == "full":
        public = pd.read_csv(registry,dtype={"corp_code":str},parse_dates=["known_since"])
        cutoff = frame.corp_code.map(public.groupby("corp_code").known_since.min())
        eligible &= cutoff.isna() | pd.to_datetime(frame.anchor).lt(cutoff)
        expanded = expanded.join(published_inputs(frame,history))
    inputs = [x.loc[eligible].reset_index(drop=True).replace([np.inf,-np.inf],np.nan) for x in (base,expanded)]
    return frame.loc[eligible,["corp_code","anchor","y_12m"]].reset_index(drop=True), inputs


def fit_one(family, x, y, codes, hard):
    w = np.ones(len(y))
    if hard:
        count = codes.map(codes.value_counts()).to_numpy()
        w /= np.sqrt(count/np.median(count))
        apparent_risk = x.impairment_ratio.rank(pct=True).fillna(.5) + (-x.roa_ttm).rank(pct=True).fillna(.5)
        w *= np.where(y==0, 1+hard*(apparent_risk.to_numpy()/2)**3, 1)
    if family == "boosting":
        model = HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=20,
                    learning_rate=.05,l2_regularization=10,early_stopping=False,random_state=SEED)
        return model.fit(x,y,sample_weight=w)
    imputer = SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)
    if family == "trees":
        model = make_pipeline(imputer,ExtraTreesClassifier(n_estimators=144,min_samples_leaf=8,
                    max_features=.8,n_jobs=2,random_state=SEED))
        return model.fit(x,y,extratreesclassifier__sample_weight=w)
    model = make_pipeline(imputer,StandardScaler(),LogisticRegression(C=.08,max_iter=2000))
    return model.fit(x,y,logisticregression__sample_weight=w)


def predict_one(model,family,x):
    return model.predict_proba(np.sign(x)*np.log1p(x.abs()) if family=="logistic" else x)[:,1]


def add_blends(scores):
    for name, parts in BLENDS.items():
        scores[name]=np.mean([scores[p] for p in parts],axis=0)


def fit(data, mode, history, registry, out):
    frame, inputs = dataset(data,mode,history,registry)
    firms = frame.groupby("corp_code").y_12m.max()
    y = frame.y_12m.to_numpy(dtype=int)
    scores = {name:np.zeros(len(y)) for name in SPEC}
    models = {name:[] for name in SPEC}
    for repeat in range(2):
        split = StratifiedKFold(4,shuffle=True,random_state=SEED+repeat)
        for step,(f,v) in enumerate(split.split(firms,firms),1):
            tr = np.flatnonzero(frame.corp_code.isin(firms.index[f]))
            va = np.flatnonzero(frame.corp_code.isin(firms.index[v]))
            for name,(family,expanded,hard) in SPEC.items():
                x=inputs[int(expanded)]
                fit_x=x.iloc[tr]
                if family=="logistic":
                    fit_x=np.sign(fit_x)*np.log1p(fit_x.abs())
                model=fit_one(family,fit_x,y[tr],frame.corp_code.iloc[tr],hard)
                models[name].append(model)
                scores[name][va]+=predict_one(model,family,x.iloc[va])/2
            print(mode, "개발 폴드", repeat*4+step, "/ 8",flush=True)
    add_blends(scores)
    policies = []
    for floor in (.80,.85,.90):
        for name,score in scores.items():
            cut = threshold_options(y,np.zeros(len(y),dtype=bool),np.ones(len(y),dtype=bool),
                                    score,int(np.ceil(floor*y.sum())))
            policies.append(dict(model=name,floor=floor,cutoff=cut,ap=average_precision_score(y,score),
                                 **measure(y,score>=cut)))
    candidates = pd.DataFrame(policies)
    winners = (candidates.sort_values(["precision","recall","ap"],ascending=False)
               .drop_duplicates("floor").sort_values("floor"))
    out.mkdir(parents=True,exist_ok=True)
    candidates.to_csv(out/"development.csv",index=False)
    lock = dict(mode=mode,train_rows=len(frame),train_companies=len(firms),train_positive=int(y.sum()),
                positive_companies=int(firms.sum()),primary=winners.iloc[0].to_dict(),
                selected=winners.to_dict("records"),policies=policies)
    (out/"lock.json").write_text(json.dumps(lock,indent=2)+"\n")
    joblib.dump(dict(models=models,companies=sorted(firms.index)),out/"models.joblib",compress=3)
    print(winners.to_string(index=False),flush=True)


def evaluate(data,mode,history,registry,out):
    frame,inputs=dataset(data,mode,history,registry)
    state=joblib.load(out/"models.joblib")
    lock=json.loads((out/"lock.json").read_text())
    if lock["mode"]!=mode or set(frame.corp_code)&set(state["companies"]):
        raise ValueError("Inconsistent target or company overlap")
    y=frame.y_12m.to_numpy(dtype=int)
    scores={name:np.mean([predict_one(m,family,inputs[int(expanded)]) for m in state["models"][name]],axis=0)
            for name,(family,expanded,_) in SPEC.items()}
    add_blends(scores)
    results=[]
    for row in lock["policies"]:
        score=scores[row["model"]]
        results.append(dict(model=row["model"],floor=row["floor"],cutoff=row["cutoff"],
                            ap=average_precision_score(y,score),**measure(y,score>=row["cutoff"])))
    report=pd.DataFrame(results)
    report.to_csv(out/"test.csv",index=False)
    for row in lock["selected"]:
        score=scores[row["model"]]
        frame[f"score_dev_{row['floor']}"]=score
        frame[f"alert_dev_{row['floor']}"]=score>=row["cutoff"]
    frame.to_csv(out/"predictions.csv.gz",index=False)
    first=lock["primary"]
    result=report.loc[report.model.eq(first["model"])&report.floor.eq(first["floor"])].iloc[0].to_dict()
    summary=dict(rows=len(y),companies=frame.corp_code.nunique(),positive=int(y.sum()),
                 positive_companies=frame.loc[y==1,"corp_code"].nunique(),primary=result,
                 fixed_policies_passing_both=int((report.recall.ge(.8)&report.precision.ge(.5)).sum()))
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("fit","test"))
    p.add_argument("mode",choices=("full","proxy35"))
    p.add_argument("data",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--history",type=Path)
    p.add_argument("--registry",type=Path)
    a=p.parse_args()
    if a.mode=="full" and (a.history is None or a.registry is None):
        p.error("full mode requires --history and --registry")
    (fit if a.phase=="fit" else evaluate)(a.data,a.mode,
        normalize_panel(a.history) if a.mode=="full" else None,a.registry,a.out)
