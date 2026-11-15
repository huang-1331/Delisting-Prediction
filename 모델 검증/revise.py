"""Ablation of rechecking strong alerts and expanding financial inputs."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cascade import (
    company_folds, fit_second, gate, measure, read, second_features, second_score, threshold_options,
)
from legacy_inputs import SEED, matrices

FAMILIES = ("logistic", "extra_trees_regularized", "boosting")
DESIGNS = {
    "protected_original": (False, False),
    "review_all_original": (True, False),
    "protected_expanded": (False, True),
    "review_all_expanded": (True, True),
}
FLOORS = (.90, .92)


def financial_additions(frame, raw):
    work = raw[["paid_cap_assets", "equity_assets", "roa_ttm", "operating_roa_ttm",
                "working_capital_assets", "current_ratio"]].copy()
    work["company"] = frame.corp_code
    work["quarter"] = pd.PeriodIndex(frame.feature_end, freq="Q").asi8
    work["anchor"] = pd.to_datetime(frame.anchor)
    by_period = work.set_index(["company", "quarter"], verify_integrity=True)
    adds = pd.DataFrame(index=work.index)
    for lag in (1, 4):
        keys = pd.MultiIndex.from_arrays([work.company, work.quarter-lag], names=by_period.index.names)
        past = by_period.reindex(keys).set_axis(work.index)
        past = past.where(past.anchor.le(work.anchor), axis=0)
        equity_change = work.equity_assets - past.equity_assets
        capital = work.paid_cap_assets.where(work.paid_cap_assets.gt(0))
        previous_capital = past.paid_cap_assets.where(past.paid_cap_assets.gt(0))
        adds[f"equity_component_{lag}q"] = -equity_change / previous_capital
        adds[f"paid_capital_component_{lag}q"] = work.equity_assets * (1/previous_capital - 1/capital)
        adds[f"equity_assets_move_{lag}q"] = equity_change
        adds[f"paid_capital_assets_move_{lag}q"] = work.paid_cap_assets - past.paid_cap_assets
        adds[f"net_roa_move_{lag}q"] = work.roa_ttm - past.roa_ttm
        adds[f"operating_roa_move_{lag}q"] = work.operating_roa_ttm - past.operating_roa_ttm
        adds[f"working_capital_move_{lag}q"] = work.working_capital_assets - past.working_capital_assets
    ratio_gap = work.current_ratio - 1
    current_liabilities = work.working_capital_assets / ratio_gap.where(ratio_gap.abs() >= .01)
    adds["implied_current_liabilities_assets"] = current_liabilities.where(current_liabilities.between(0, 3))
    adds["current_ratio_shortfall"] = (1-work.current_ratio).clip(lower=0)
    adds["operating_net_income_gap"] = work.operating_roa_ttm-work.roa_ttm
    adds["capital_only_impairment_increase"] = (
        adds.paid_capital_component_1q.gt(0) & adds.equity_assets_move_1q.ge(0)
    ).astype(float).where(adds.paid_capital_component_1q.notna() & adds.equity_assets_move_1q.notna())
    adds["recovery_without_equity_erosion"] = (
        adds.net_roa_move_1q.gt(0) & adds.operating_roa_move_1q.gt(0) & adds.equity_assets_move_1q.ge(0)
    ).astype(float).where(adds[["net_roa_move_1q", "operating_roa_move_1q", "equity_assets_move_1q"]].notna().all(axis=1))
    return adds.reindex(frame.index).replace([np.inf, -np.inf], np.nan)


def specialist_inputs(raw, additions, score, cut, ids, expanded):
    x = second_features(raw.iloc[ids], score, cut)
    return x.join(additions.iloc[ids]) if expanded else x


def develop(train, raw, additions, baseline):
    y = train.y_12m.to_numpy()
    first_cut = baseline["first_cut"]
    oof_gate = np.zeros(len(y))
    configs = [(expand, family) for expand in (False, True) for family in FAMILIES]
    scores = {c: np.zeros(len(y)) for c in configs}
    models = {c: [] for c in configs}
    for repeat in range(2):
        for outer, (fit, dev) in enumerate(company_folds(train, SEED + repeat, 4)):
            primary = baseline["gate_models"][repeat*4 + outer]
            dev_score = primary.predict_proba(raw.iloc[dev])[:, 1]
            oof_gate[dev] += dev_score / 2
            inner = np.zeros(len(fit))
            for build, val in company_folds(train.iloc[fit], SEED + 11 + outer + 4*repeat, 3):
                local_gate = gate(raw.iloc[fit[build]], y[fit[build]])
                inner[val] = local_gate.predict_proba(raw.iloc[fit[val]])[:, 1]
            use = inner >= .5*first_cut
            ids = fit[use]
            for expand in (False, True):
                x = specialist_inputs(raw, additions, inner[use], first_cut, ids, expand)
                v = specialist_inputs(raw, additions, dev_score, first_cut, dev, expand)
                for family in FAMILIES:
                    key = (expand, family)
                    specialist = fit_second(family, x, y[ids], train.corp_code.iloc[ids])
                    scores[key][dev] += second_score(specialist, family, v) / 2
                    models[key].append(specialist)
            print("개발 폴드", repeat*4 + outer + 1, "/ 8", flush=True)
    return oof_gate, scores, models


def fit(train_path, baseline_path, out):
    train = read(train_path)
    baseline = joblib.load(baseline_path)
    if set(train.corp_code) != set(baseline["train_companies"]):
        raise ValueError("The frozen first stage was trained on different companies")
    raw = matrices(train)["trajectory"]
    additions = financial_additions(train, raw)
    first, scores, models = develop(train, raw, additions, baseline)
    y = train.y_12m.to_numpy()
    first_alert = first >= baseline["first_cut"]
    original_protected = first >= baseline["policy"]["guard"]
    policies = []
    for design, (review_all, expanded) in DESIGNS.items():
        protected = np.zeros(len(y), dtype=bool) if review_all else original_protected
        review = first_alert & ~protected
        for floor in FLOORS:
            min_hits = max(int(np.ceil(floor*y.sum())), int(np.ceil(.95*y[first_alert].sum())))
            for family in FAMILIES:
                score = scores[expanded, family]
                cut = threshold_options(y, protected, review, score, min_hits)
                alerts = protected | (review & (score >= cut))
                policies.append(dict(design=design, expanded=expanded, review_all=review_all,
                                     family=family, floor=floor, cutoff=cut, **measure(y, alerts)))
    table = pd.DataFrame(policies)
    winners = (table.sort_values(["precision", "recall"], ascending=False)
               .drop_duplicates(["design", "floor"]).sort_values(["floor", "design"]))
    overall = winners[winners.floor.eq(.90)].sort_values(["precision", "recall"], ascending=False).iloc[0]
    old = table[(table.design=="protected_original") & (table.floor==.90) & (table.family=="logistic")].iloc[0]
    np.testing.assert_allclose(old.cutoff, baseline["policy"]["second_cut"], atol=1e-13, rtol=0)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out/"development.csv", index=False)
    (out/"lock.json").write_text(json.dumps(dict(primary=overall.to_dict(),
                                         by_design=winners.to_dict("records")), indent=2)+"\n")
    selected_keys = {(bool(r.expanded), r.family) for r in winners.itertuples()}
    joblib.dump(dict(baseline=baseline, models={key: models[key] for key in selected_keys}), out/"models.joblib", compress=3)
    print(winners.to_string(index=False), flush=True)


def score_specialist(raw, additions, gates, models, cut, review, policy):
    ids = np.flatnonzero(review)
    return np.mean([
        second_score(model, policy["family"], specialist_inputs(raw, additions, g[ids], cut, ids, policy["expanded"]))
        for g, model in zip(gates, models, strict=True)
    ], axis=0)


def evaluate(test_path, baseline_predictions, out):
    test = read(test_path)
    state = joblib.load(out/"models.joblib")
    baseline = state["baseline"]
    lock = json.loads((out/"lock.json").read_text())
    old = pd.read_csv(baseline_predictions, dtype={"corp_code": str})
    if set(test.corp_code) & set(baseline["train_companies"]) or not test[["corp_code", "anchor", "y_12m"]].equals(old[["corp_code", "anchor", "y_12m"]]):
        raise ValueError("Test cohort differs from the company-disjoint benchmark")
    raw = matrices(test)["trajectory"]
    extra = financial_additions(test, raw)
    gates = [model.predict_proba(raw)[:, 1] for model in baseline["gate_models"]]
    risk = np.mean(gates, axis=0)
    first = risk >= baseline["first_cut"]
    strong = risk >= baseline["policy"]["guard"]
    np.testing.assert_array_equal(first, old.first_alert)
    np.testing.assert_array_equal(strong, old.protected)
    y = test.y_12m.to_numpy()
    output = old[["corp_code", "anchor", "y_12m", "first_alert", "protected", "final_alert"]].copy()
    rows = [dict(design="first_stage", **measure(y, first)), dict(design="previous_logistic", **measure(y, old.final_alert.to_numpy()))]
    for policy in lock["by_design"]:
        protected = np.zeros(len(y), dtype=bool) if policy["review_all"] else strong
        review = first & ~protected
        models = state["models"][policy["expanded"], policy["family"]]
        specialist = score_specialist(raw, extra, gates, models, baseline["first_cut"], review, policy)
        alert = protected.copy()
        alert[review] = specialist >= policy["cutoff"]
        removed_strong = strong & ~alert
        rows.append(dict(design=policy["design"], family=policy["family"], floor=policy["floor"],
                         reviewed=int(review.sum()), strong_fp_removed=int((removed_strong & (y==0)).sum()),
                         strong_tp_removed=int(y[removed_strong].sum()), **measure(y, alert)))
        output[f"{policy['design']}_{policy['floor']}_alert"] = alert
    pd.DataFrame(rows).to_csv(out/"test.csv", index=False)
    output.to_csv(out/"predictions.csv.gz", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "test"))
    for name in ("data", "baseline", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    (fit if args.phase=="fit" else evaluate)(args.data, args.baseline, args.out)
