"""Twelve-month landmark models with company-level development separation."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 38271
RECALL_FLOOR = 0.85
FEATURES = """
impairment_ratio impairment_delta_1q impairment_mean_4q equity_delta_4q_assets
debt_assets debt_assets_delta_1q current_ratio working_capital_assets roa_ttm
operating_roa_ttm loss_fraction_4q retained_assets retained_deficit_to_capital
erosion_pressure_at_label_limit projected_impairment liability_to_capital loss_to_capital_proxy
""".split()


def load_snapshots(path):
    frame = pd.read_csv(path, dtype={"corp_code": str},
                        parse_dates=["anchor", "feature_end", "event_date", "followup_end"])
    if frame.duplicated(["corp_code", "anchor"]).any() or (frame.feature_end > frame.anchor).any():
        raise ValueError("Duplicate snapshot or financial period after prediction date")
    return frame


def quarters(series):
    return series.dt.year * 4 + series.dt.quarter


def duration_and_event(frame):
    duration = quarters(frame.followup_end) - quarters(frame.anchor)
    event = quarters(frame.event_date) - quarters(frame.anchor)
    if (duration.isna() | (duration < 0) | ((event <= 0) | (event > duration))).any():
        raise ValueError("Invalid event or follow-up date")
    return duration, event


def inventory(frame):
    duration, event = duration_and_event(frame)
    within = event.between(1, 4)
    negative = ~within & duration.ge(4)
    return dict(rows=len(frame), companies=frame.corp_code.nunique(),
                event_rows=int(within.sum()), event_companies=frame.loc[within, "corp_code"].nunique(),
                quarter_events=[int(event.eq(i).sum()) for i in range(1, 5)],
                full_negative=int(negative.sum()), censored_early=int((~within & duration.between(1, 3)).sum()),
                no_followup=int(duration.eq(0).sum()))


def features(frame):
    return frame[FEATURES].replace([np.inf, -np.inf], np.nan)


def hazard_inputs(frame, quarter):
    data = features(frame).copy()
    for step in (2, 3, 4):
        data[f"step_{step}"] = (quarter == step).astype(float)
    return data


def hazard_fit(frame):
    duration, event = duration_and_event(frame)
    horizon = np.minimum(np.minimum(duration.to_numpy(), event.fillna(np.inf).to_numpy()), 4).astype(int)
    row, step = np.where(np.arange(1, 5)[None, :] <= horizon[:, None])
    step += 1
    y = (event.to_numpy()[row] == step).astype(int)
    model = logistic()
    model.fit(hazard_inputs(frame.iloc[row], step), y)
    return model, len(y)


def logistic():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler(), LogisticRegression(max_iter=2000))


def hazard_predict(model, frame):
    conditional = np.column_stack([
        model.predict_proba(hazard_inputs(frame, np.full(len(frame), step)))[:, 1]
        for step in range(1, 5)
    ])
    return 1 - np.cumprod(1 - conditional, axis=1), conditional


def threshold(y, score):
    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(order) - 1]
    hits = np.cumsum(y[order])[ends]
    possible = np.flatnonzero(hits >= RECALL_FLOOR * y.sum())
    if not len(possible) or y.sum() == 0:
        raise ValueError("Cannot set recall target on development data")
    best = max(possible, key=lambda i: (hits[i] / (ends[i] + 1), hits[i], sorted_scores[ends[i]]))
    return float(sorted_scores[ends[best]])


def metrics(y, score, cut):
    alerts = score >= cut
    tp = int(y[alerts].sum())
    fp = int(alerts.sum() - tp)
    return dict(tp=tp, fp=fp, fn=int(y.sum()-tp), alerts=tp+fp,
                recall=tp/y.sum(), precision=tp/(tp+fp) if tp+fp else None,
                ap=average_precision_score(y, score), threshold=cut)


def archived_tabpfn(path, test_positives):
    with ZipFile(path) as archive:
        lock = json.loads(archive.read("opt/lock.json"))
        with archive.open("opt/test_summary.csv") as f:
            candidates = pd.read_csv(f)
    selected = max((r for r in lock["all"] if r["recall"] >= RECALL_FLOOR),
                   key=lambda r: (r["precision"], r["recall"]))
    row = candidates.loc[
        candidates["name"].eq(selected["name"]) & candidates.goal.eq(selected["goal"])
        & np.isclose(candidates.threshold, selected["threshold"])
    ].iloc[0].to_dict()
    row["fn"] = test_positives - row["tp"]
    row["alerts"] = row["tp"] + row["fp"]
    return {"development": selected, "test": row}


def run(train_path, test_path, tabpfn_path, out):
    out.mkdir(parents=True, exist_ok=True)
    train = load_snapshots(train_path)
    test = load_snapshots(test_path)
    if set(train.corp_code) & set(test.corp_code):
        raise ValueError("Company overlap in train and test")
    for frame in (train, test):
        duration, event = duration_and_event(frame)
        if not np.array_equal(event.between(1, 4).astype(int), frame.y_12m):
            raise ValueError("The timing labels disagree with the legacy 12-month labels")
        if (event.isna() & duration.lt(4)).any():
            raise ValueError("Censored snapshots cannot use the legacy binary label")
    groups = train.groupby("corp_code")[["y_12m", "y_24m"]].max()
    split = StratifiedKFold(4, shuffle=True, random_state=SEED)
    fit_ids, dev_ids = next(split.split(groups, groups.y_12m * 2 + groups.y_24m))
    fit = train[train.corp_code.isin(groups.index[fit_ids])]
    dev = train[train.corp_code.isin(groups.index[dev_ids])]
    y_dev, y_test = dev.y_12m.to_numpy(), test.y_12m.to_numpy()

    hazard, at_risk_rows = hazard_fit(fit)
    binary = logistic().fit(features(fit), fit.y_12m)
    trees = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          ExtraTreesClassifier(n_estimators=105, min_samples_leaf=4,
                                               max_features=.85, random_state=SEED, n_jobs=2))
    trees.fit(features(fit), fit.y_12m)
    models = {"quarterly_hazard": hazard, "binary_logistic": binary, "extra_trees": trees}
    cutoffs = {}
    development = {}
    for name, model in models.items():
        scores = hazard_predict(model, dev)[0][:, -1] if name == "quarterly_hazard" else model.predict_proba(features(dev))[:, 1]
        cutoffs[name] = threshold(y_dev, scores)
        development[name] = metrics(y_dev, scores, cutoffs[name])

    joblib.dump(dict(target="first_observed_impairment_over_35_percent", features=FEATURES,
                     models=models, thresholds=cutoffs), out / "model.joblib")
    cumulative, conditional = hazard_predict(hazard, test)
    predictions = test[["corp_code", "anchor", "feature_end", "event_date", "followup_end", "y_12m"]].copy()
    for index in range(4):
        predictions[f"conditional_risk_q{index+1}"] = conditional[:, index]
        predictions[f"cumulative_risk_q{index+1}"] = cumulative[:, index]
    comparison = {}
    for name, model in models.items():
        scores = cumulative[:, -1] if name == "quarterly_hazard" else model.predict_proba(features(test))[:, 1]
        predictions[f"{name}_score"] = scores
        predictions[f"{name}_alert"] = scores >= cutoffs[name]
        comparison[name] = dict(development=development[name], test=metrics(y_test, scores, cutoffs[name]))
    if tabpfn_path:
        comparison["archived_tabpfn"] = archived_tabpfn(tabpfn_path, int(y_test.sum()))
    report = dict(target="first_observed_impairment_over_35_percent", training=inventory(train),
                  fit=inventory(fit), development=inventory(dev), test=inventory(test),
                  at_risk_fit_rows=at_risk_rows, comparison=comparison)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    predictions.to_csv(out / "predictions.csv.gz", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--tabpfn-archive", type=Path)
    args = parser.parse_args()
    run(args.train, args.test, args.tabpfn_archive, args.out)
