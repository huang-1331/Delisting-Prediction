"""Retain strong alerts and re-evaluate only the first-stage boundary."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from legacy_inputs import SEED, matrices

STAGE2_DEV_RECALL = .90
STAGE2_HIT_RETENTION = .95
FAMILIES = ("logistic", "extra_trees", "extra_trees_regularized", "boosting")


def read(path):
    frame = pd.read_csv(path, dtype={"corp_code": str})
    if frame.duplicated(["corp_code", "anchor"]).any() or frame.feature_end.gt(frame.anchor).any():
        raise ValueError("Invalid company-quarter snapshots")
    return frame


def company_folds(frame, seed, n_splits):
    groups = frame.groupby("corp_code")
    strata = groups.y_12m.max()
    if "y_24m" in frame:
        strata = 2 * strata + groups.y_24m.max()
    companies = frame.corp_code
    split = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for fit, dev in split.split(strata, strata):
        yield (np.flatnonzero(companies.isin(strata.index[fit])),
               np.flatnonzero(companies.isin(strata.index[dev])))


def gate(frame, labels):
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          ExtraTreesClassifier(n_estimators=105, min_samples_leaf=4, max_features=.85,
                                               n_jobs=2, random_state=SEED))
    return model.fit(frame, labels, extratreesclassifier__sample_weight=np.ones(len(labels)))


def second_features(raw, gate_score, cut):
    cols = [
        "impairment_ratio", "impairment_delta_1q", "impairment_slope", "debt_assets",
        "debt_assets_delta_1q", "current_ratio", "working_capital_assets", "roa_ttm",
        "operating_roa_ttm", "loss_fraction_4q", "equity_delta_4q_assets", "retained_assets",
        "loss_to_capital_proxy", "net_loss_cover", "roa_ttm_slope", "operating_roa_ttm_slope",
        "working_capital_assets_slope", "debt_assets_slope", "impairment_ratio_accel",
    ]
    signals = raw[cols].copy()
    signals["income_equity_joint_weakness"] = (
        raw.roa_ttm.lt(0) & raw.equity_delta_4q_assets.lt(0)
    ).astype(float).where(raw.roa_ttm.notna() & raw.equity_delta_4q_assets.notna())
    signals["earnings_rebound"] = (
        raw.roa_ttm_slope.gt(0) & raw.operating_roa_ttm_slope.gt(0)
    ).astype(float).where(raw.roa_ttm_slope.notna() & raw.operating_roa_ttm_slope.notna())
    signals["gate_score_relative_to_cut"] = gate_score / max(cut, .001)
    return signals


def fit_second(family, x, y, companies):
    counts = companies.map(companies.value_counts()).to_numpy()
    weights = 1 / np.sqrt(counts)
    weights /= weights.mean()
    if family == "logistic":
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              StandardScaler(), LogisticRegression(C=.08, max_iter=1500))
        model.fit(np.sign(x) * np.log1p(x.abs()), y, logisticregression__sample_weight=weights)
    elif family in ("extra_trees", "extra_trees_regularized"):
        leaf = 6 if family == "extra_trees" else 18
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              ExtraTreesClassifier(n_estimators=200, min_samples_leaf=leaf,
                                                   max_features=.85, n_jobs=2, random_state=SEED))
        model.fit(x, y, extratreesclassifier__sample_weight=weights)
    else:
        model = HistGradientBoostingClassifier(max_iter=110, learning_rate=.045, max_leaf_nodes=7,
                                                min_samples_leaf=25, l2_regularization=12,
                                                early_stopping=False, random_state=SEED)
        model.fit(x, y, sample_weight=weights)
    return model


def second_score(model, family, x):
    if family == "logistic":
        x = np.sign(x) * np.log1p(x.abs())
    return model.predict_proba(x)[:, 1]


def development_predictions(frame, raw, cut, first_stage_models=None):
    y = frame.y_12m.to_numpy()
    oof_gate = np.zeros(len(y))
    oof_second = {family: np.zeros(len(y)) for family in FAMILIES}
    gate_models = []
    training_cuts = []
    second_models = {family: [] for family in FAMILIES}
    for repeat in range(2):
        for outer, (fit, dev) in enumerate(company_folds(frame, SEED + repeat, 4)):
            model = (gate(raw.iloc[fit], y[fit]) if first_stage_models is None
                     else first_stage_models[len(gate_models)])
            gate_models.append(model)
            dev_gate = model.predict_proba(raw.iloc[dev])[:, 1]
            oof_gate[dev] += dev_gate / 2

            inner_gate = np.zeros(len(fit))
            for build, score in company_folds(frame.iloc[fit], SEED + 11 + outer + 4 * repeat, 3):
                inner = gate(raw.iloc[fit[build]], y[fit[build]])
                inner_gate[score] = inner.predict_proba(raw.iloc[fit[score]])[:, 1]
            local_cut = cut if cut is not None else primary_cut(y[fit], inner_gate)
            training_cuts.append(local_cut)
            # Train on the neighbourhood of the existing cut, with nearby below-cut rows as context.
            close = np.flatnonzero(inner_gate >= .5 * local_cut)
            fit_ids = fit[close]
            x_fit = second_features(raw.iloc[fit_ids], inner_gate[close], local_cut)
            x_dev = second_features(raw.iloc[dev], dev_gate, local_cut)
            for family in FAMILIES:
                specialist = fit_second(family, x_fit, y[fit_ids], frame.corp_code.iloc[fit_ids])
                second_models[family].append(specialist)
                oof_second[family][dev] += second_score(specialist, family, x_dev) / 2
            print(f"Development split {len(gate_models)} / 8", flush=True)
    return oof_gate, oof_second, gate_models, second_models, training_cuts


def measure(y, warning):
    hits = int(y[warning].sum())
    n = int(warning.sum())
    total = int(y.sum())
    return dict(tp=hits, fp=n-hits, fn=total-hits, alerts=n,
                recall=hits/total, precision=hits/n if n else None)


def threshold_options(y, protected, review, specialist, min_hits):
    ranked = np.flatnonzero(review)
    ranked = ranked[np.argsort(-specialist[ranked], kind="stable")]
    scores = specialist[ranked]
    # Option 0 keeps only protected alerts; the remaining choices include every tied score.
    ends = np.r_[0, np.flatnonzero(scores[:-1] != scores[1:])+1, len(ranked)]
    if not len(ranked):
        ends = np.array([0])
    hits = int(y[protected].sum()) + np.r_[0, np.cumsum(y[ranked])][ends]
    alerts = int(protected.sum()) + ends
    allowed = np.flatnonzero(hits >= min_hits)
    if not len(allowed):
        return None
    i = max(allowed, key=lambda j: (hits[j]/alerts[j], hits[j], -alerts[j]))
    return float(scores[ends[i]-1]) if ends[i] else float(np.nextafter(scores.max(), np.inf)) if len(scores) else 1.0


def primary_cut(y, score):
    return threshold_options(y, np.zeros(len(y), dtype=bool), np.ones(len(y), dtype=bool),
                             score, int(np.ceil(.93*y.sum())))


def select_policy(y, score, stage2, cut):
    stage1 = score >= cut
    first = measure(y, stage1)
    min_hits = max(int(np.ceil(STAGE2_DEV_RECALL*y.sum())),
                   int(np.ceil(STAGE2_HIT_RETENTION*first["tp"])))
    candidates = []
    for quantile in (.50, .65, .75):
        guard = float(np.quantile(score[stage1], quantile))
        protected = score >= guard
        review = stage1 & ~protected
        if not review.any():
            continue
        for family, second in stage2.items():
            second_cut = threshold_options(y, protected, review, second, min_hits)
            if second_cut is None:
                continue
            kept = protected | (review & (second >= second_cut))
            candidates.append(dict(family=family, guard=guard, guard_quantile=quantile,
                                   second_cut=second_cut, reviewed=int(review.sum()), **measure(y, kept)))
    if not candidates:
        raise ValueError("No second-stage development candidate")
    chosen = max(candidates, key=lambda p: (p["precision"], p["recall"], -p["reviewed"]))
    if chosen["precision"] <= first["precision"]:
        chosen = dict(family=chosen["family"], guard=cut, guard_quantile=0.0,
                      second_cut=0.0, reviewed=0, **first)
    standalone_cut = threshold_options(y, np.zeros(len(y), dtype=bool), stage1, score, min_hits)
    return chosen, standalone_cut, candidates


def predict_cascade(bundle, raw):
    base_scores = [model.predict_proba(raw)[:, 1] for model in bundle["gate_models"]]
    score = np.mean(base_scores, axis=0)
    stage1 = score >= bundle["first_cut"]
    policy = bundle["policy"]
    protected = score >= policy["guard"]
    review = stage1 & ~protected
    second = np.full(len(score), np.nan)
    if review.any():
        ids = np.flatnonzero(review)
        family = policy["family"]
        second[ids] = np.mean([
            second_score(model, family, second_features(raw.iloc[ids], values[ids], local_cut))
            for model, values, local_cut in zip(bundle["second_models"], base_scores, bundle["training_cuts"], strict=True)
        ], axis=0)
    final = protected | (review & (second >= policy["second_cut"]))
    if np.any(final & ~stage1) or np.any(protected & ~final):
        raise ValueError("Cascade violated the alert boundary")
    return score, stage1, protected, review, second, final


def fit(train_path, prior_path, out):
    frame = read(train_path)
    with ZipFile(prior_path) as archive:
        previous = json.loads(archive.read("out/lock.json"))
    if hashlib.sha256(train_path.read_bytes()).hexdigest() != previous["train_sha"]:
        raise ValueError("Training data differs from the frozen first-stage experiment")
    prior = previous["winners"]["12-0.92"]
    if prior["model"] != "results/trajectory_extra_4_weight1":
        raise ValueError("Unexpected first-stage model")
    cut = prior["threshold"]
    raw = matrices(frame)["trajectory"]
    score, stage2, gate_models, second_models, training_cuts = development_predictions(frame, raw, cut)
    y = frame.y_12m.to_numpy()
    first = measure(y, score >= cut)
    if first["tp"] != prior["tp"] or first["fp"] != prior["fp"]:
        raise ValueError(f"First-stage development counts do not match prior run: {first}")
    selected, simple, candidates = select_policy(y, score, stage2, cut)
    bundle = dict(first_cut=cut, policy=selected, standalone_cut=simple, gate_models=gate_models,
                  training_cuts=training_cuts,
                  second_models=second_models[selected["family"]], train_companies=sorted(frame.corp_code.unique()))
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "cascade.joblib", compress=3)
    development = dict(first_stage=first, two_stage=selected, single_cut=measure(y, score >= simple))
    (out / "cascade_lock.json").write_text(json.dumps(development, indent=2) + "\n")
    pd.DataFrame(candidates).to_csv(out / "cascade_development.csv", index=False)
    print(json.dumps(development, indent=2), flush=True)


def evaluate(test_path, prior_path, out):
    frame = read(test_path)
    bundle = joblib.load(out / "cascade.joblib")
    if set(frame.corp_code) & set(bundle["train_companies"]):
        raise ValueError("Training and test companies overlap")
    score, stage1, protected, review, second, final = predict_cascade(bundle, matrices(frame)["trajectory"])
    with ZipFile(prior_path) as archive, archive.open("out/predictions.csv.gz") as f:
        expected = pd.read_csv(f, compression="gzip", dtype={"corp_code": str})
    if not frame[["corp_code", "anchor"]].equals(expected[["corp_code", "anchor"]]):
        raise ValueError("The test sample does not match the frozen first-stage predictions")
    np.testing.assert_allclose(score, expected["12_0.92_score"], atol=1e-13, rtol=0)
    if not np.array_equal(stage1, expected["12_0.92_alert"]):
        raise ValueError("First-stage warnings do not match the frozen benchmark")
    y = frame.y_12m.to_numpy()
    report = dict(first_stage=measure(y, stage1), two_stage=measure(y, final),
                  single_cut=measure(y, score >= bundle["standalone_cut"]),
                  protected=int(protected.sum()), reviewed=int(review.sum()),
                  removed_tp=int(y[stage1 & ~final].sum()), removed_fp=int((stage1 & ~final & (y==0)).sum()))
    (out / "cascade_test.json").write_text(json.dumps(report, indent=2) + "\n")
    frame[["corp_code", "anchor", "y_12m"]].assign(first_score=score, first_alert=stage1, protected=protected,
                                                   reviewed=review, second_score=second, final_alert=final
                                                   ).to_csv(out / "cascade_predictions.csv.gz", index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "test"))
    parser.add_argument("data", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    (fit if args.phase == "fit" else evaluate)(args.data, args.prior, args.out)
