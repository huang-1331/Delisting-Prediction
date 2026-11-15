"""Re-fit the supplied MLP architectures for 12-month complete impairment."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cascade import measure, threshold_options
from final_model import dataset
from full_impairment import normalize_panel

LAYERS = ((16, 8), (32, 16, 8), (32, 32, 16, 8))
CONFIGS = {("MLP_" + "_".join(map(str, layers)) + ("_weighted" if weighted else "")):
           (layers, weighted) for layers in LAYERS for weighted in (False, True)}
SEEDS = (42, 43, 44)


def fit_models(x, y, config):
    layers, weighted = config
    weights = np.ones(len(y))
    if weighted:
        weights[y == 1] = min(15, np.count_nonzero(y == 0) / y.sum())
        weights /= weights.mean()
    models = []
    for seed in SEEDS:
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=layers, alpha=.02, max_iter=350,
                          batch_size=64, random_state=seed, early_stopping=False))
        models.append(model.fit(x, y, mlpclassifier__sample_weight=weights))
    return models


def score(models, x):
    return np.mean([p.predict_proba(x)[:, 1] for p in models], axis=0)


def features(path, history, registry):
    frame, (x, _) = dataset(path, "full", history, registry)
    prohibited = {"anchor", "event_date", "followup_end", "last_observed_quarter", "corp_code"}
    assert not (set(x) & prohibited) and not any(col.startswith("y_") for col in x)
    return frame, x


def fit(path, history, registry, out):
    frame, x = features(path, history, registry)
    y = frame.y_12m.to_numpy(dtype=int)
    firms = frame.groupby("corp_code").y_12m.max()
    validation = {name: np.zeros(len(y)) for name in CONFIGS}
    for seed in SEEDS:
        folds = StratifiedKFold(3,shuffle=True,random_state=seed)
        for tr, va in folds.split(firms, firms):
            train = frame.corp_code.isin(firms.index[tr]).to_numpy()
            valid = frame.corp_code.isin(firms.index[va]).to_numpy()
            assert not (train & valid).any()
            for name, config in CONFIGS.items():
                models = fit_models(x.loc[train], y[train], config)
                validation[name][valid] += score(models, x.loc[valid]) / len(SEEDS)
    policies = {}
    for name, s in validation.items():
        cut = threshold_options(y, np.zeros(len(y),dtype=bool), np.ones(len(y),dtype=bool),
                                s, int(np.ceil(.8*y.sum())))
        policies[name] = dict(cutoff=cut, ap=average_precision_score(y,s), **measure(y,s>=cut))
    primary = max(policies, key=lambda n: (policies[n]["precision"], policies[n]["ap"]))
    out.mkdir(parents=True, exist_ok=True)
    state = dict(policies=policies, primary=primary, train_rows=len(y), positive=int(y.sum()),
                 companies=sorted(firms.index), event_companies=int(firms.sum()))
    (out/"lock.json").write_text(json.dumps(state,indent=2)+"\n")
    fitted = {name:fit_models(x,y,config) for name,config in CONFIGS.items()}
    joblib.dump(fitted,out/"models.joblib",compress=3)
    print("개발 선정",primary,policies[primary],flush=True)


def test(path, history, registry, out, prior, boosting):
    frame, x = features(path,history,registry)
    y = frame.y_12m.to_numpy(dtype=int)
    lock = json.loads((out/"lock.json").read_text())
    assert not set(lock["companies"]) & set(frame.corp_code)
    archive = joblib.load(out/"models.joblib")
    metrics = []
    for name, models in archive.items():
        s = score(models,x)
        warning = s >= lock["policies"][name]["cutoff"]
        metrics.append(dict(model=name, ap=average_precision_score(y,s), **measure(y,warning)))
        frame[name+"_score"]=s
        frame[name+"_warning"]=warning
    keys = ["corp_code","anchor","y_12m"]
    for name,path,column in (("순손실",prior,"annual_loss_projection_0.8_alert"),
                             ("순손실+부스팅",boosting,"final_alert_0.9")):
        old = pd.read_csv(path,dtype={"corp_code":str}).rename(columns={"y_12": "y_12m"})
        pd.testing.assert_frame_equal(old[keys],frame[keys],check_dtype=False)
        metrics.append(dict(model=name, **measure(y,old[column].to_numpy())))
        frame[name+"_warning"]=old[column]
    table=pd.DataFrame(metrics)
    table.to_csv(out/"test.csv",index=False)
    frame.to_csv(out/"predictions.csv.gz",index=False)
    print(table.to_string(index=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=("fit","test"))
    for arg in ("data","history","registry","out"):
        p.add_argument(arg,type=Path)
    p.add_argument("--prior",type=Path)
    p.add_argument("--boosting",type=Path)
    a=p.parse_args()
    args=(a.data,normalize_panel(a.history),a.registry,a.out)
    if a.phase=="fit":
        fit(*args)
    else:
        test(*args,a.prior,a.boosting)
