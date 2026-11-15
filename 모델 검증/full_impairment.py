"""Build and evaluate an OFS equity <= 0 retrospective pilot."""
import argparse
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from algorithm5 import (
    FEATURES, SEED, duration_and_event, features, hazard_fit, hazard_predict, inventory,
    load_snapshots, logistic, metrics, threshold,
)


def available_from_receipt(value):
    return max(re.findall(r"\b(20\d{6})\d{6}\b", str(value)), default="")


def normalize_panel(path, fresh=False):
    panel = pd.read_csv(path, dtype={"corp_code": str, "report_code": str}, low_memory=False)
    panel["period_end"] = pd.to_datetime(panel.period_end, errors="coerce")
    if fresh:
        panel["available"] = pd.to_datetime(panel.available, errors="coerce")
    else:
        panel["available"] = pd.to_datetime(panel.receipt_numbers.map(available_from_receipt),
                                            format="%Y%m%d", errors="coerce")
        panel = panel.loc[panel.usable.eq(True) & panel.balance_reconciles.eq(True)
                          & panel.fs_div.eq("OFS") & panel.currency.eq("KRW")]
    panel = panel.loc[panel.available.ge(panel.period_end) & panel.assets.gt(0)
                      & panel.equity.notna() & panel.period_end.dt.is_quarter_end].copy()
    panel["quarter"] = panel.period_end.dt.to_period("Q").astype(int)
    if panel.duplicated(["corp_code", "quarter"]).any():
        raise ValueError("Ambiguous company-quarter accounting observations")
    return panel


def financial_features(history):
    period = history.period_end
    annual = period.dt.month.eq(12)
    adjacent = history.index.to_series().diff().eq(1)
    four = adjacent.rolling(3, min_periods=3).sum().eq(3)
    five = adjacent.rolling(4, min_periods=4).sum().eq(4)
    for field in ("net_income", "operating_income"):
        current = history[field].where(~annual)
        previous_three = current.rolling(3, min_periods=3).sum().shift()
        standalone = current.where(~annual, (history[field] - previous_three).where(four))
        history[field + "_standalone"] = standalone
        history[field + "_ttm"] = history[field].where(annual,
                                                        standalone.rolling(4, min_periods=4).sum().where(four))
    assets, capital = history.assets, history.capital.where(history.capital.gt(0))
    impairment = 1 - history.equity / capital
    impairment_delta = impairment.diff().where(adjacent)
    debt_assets = history.liabilities / assets
    net_income = history.net_income_ttm
    standalone = history.net_income_standalone
    result = dict(
        impairment_ratio=impairment,
        impairment_delta_1q=impairment_delta,
        impairment_mean_4q=impairment.rolling(4, min_periods=4).mean().where(four),
        equity_delta_4q_assets=((history.equity-history.equity.shift(4))/assets.shift(4)).where(five),
        debt_assets=debt_assets,
        debt_assets_delta_1q=debt_assets.diff().where(adjacent),
        current_ratio=history.current_assets / history.current_liabilities.where(history.current_liabilities.gt(0)),
        working_capital_assets=(history.current_assets-history.current_liabilities)/assets,
        roa_ttm=net_income/assets,
        operating_roa_ttm=history.operating_income_ttm/assets,
        loss_fraction_4q=standalone.lt(0).astype(float).where(standalone.notna()).rolling(4, min_periods=4).mean().where(four),
        retained_assets=history.retained_earnings/assets,
        retained_deficit_to_capital=(-history.retained_earnings/capital).clip(lower=0),
        erosion_pressure_at_label_limit=impairment_delta.clip(lower=0)/((.35-impairment).clip(lower=0)+.1),
        projected_impairment=impairment+2*impairment_delta.clip(lower=0).fillna(0),
        liability_to_capital=history.liabilities/capital,
        loss_to_capital_proxy=-net_income/capital,
    )
    return {key: value.iloc[-1] for key, value in result.items()}


def snapshots(panel, include_features):
    rows = []
    for corp, group in panel.groupby("corp_code"):
        group = group.set_index("quarter").sort_index()
        observed = set(group.index)
        equity = group.equity.to_dict()
        first_observed = group.loc[group.equity.le(0), "period_end"].min()
        last_observed = group.period_end.iloc[-1]
        for feature_q, row in group.iterrows():
            anchor = row.period_end + pd.offsets.MonthEnd(6)
            anchor_q = feature_q + 2
            if (row.available > anchor or not row.capital > 0
                    or not set(range(feature_q, anchor_q + 1)).issubset(observed)
                    or (pd.notna(first_observed) and first_observed <= anchor)):
                continue
            event = pd.NaT
            stop_q = anchor_q
            for future in range(anchor_q + 1, anchor_q + 5):
                if future not in observed:
                    break
                stop_q = future
                if equity[future] <= 0:
                    event = group.at[future, "period_end"]
                    break
            record = dict(corp_code=corp, feature_end=row.period_end, anchor=anchor, event_date=event,
                          followup_end=group.at[stop_q, "period_end"],
                          first_observed_equity_nonpositive=first_observed, last_observed_quarter=last_observed)
            if include_features:
                history = group.loc[:feature_q].loc[lambda frame: frame.available.le(anchor)].copy()
                record.update(financial_features(history))
            rows.append(record)
    frame = pd.DataFrame(rows)
    duration, event = duration_and_event(frame)
    frame["y_12m"] = np.where(event.between(1, 4), 1, np.where(duration.ge(4), 0, np.nan))
    return frame


def run(args):
    tr_ids = set(load_snapshots(args.train).corp_code)
    te_ids = set(load_snapshots(args.test).corp_code)
    if tr_ids & te_ids:
        raise ValueError("Company overlap in original split")
    panel = normalize_panel(args.history)
    frame = snapshots(panel.loc[panel.corp_code.isin(tr_ids | te_ids)], include_features=True)
    train = frame.loc[frame.corp_code.isin(tr_ids)]
    test_all = frame.loc[frame.corp_code.isin(te_ids)]
    test = test_all.dropna(subset=["y_12m"])
    groups = train.groupby("corp_code").y_12m.max().fillna(0)
    split = StratifiedKFold(4, shuffle=True, random_state=SEED)
    fit_ids, dev_ids = next(split.split(groups, groups))
    fit_all = train.loc[train.corp_code.isin(groups.index[fit_ids])]
    fit = fit_all.dropna(subset=["y_12m"])
    dev_all = train.loc[train.corp_code.isin(groups.index[dev_ids])]
    dev = dev_all.dropna(subset=["y_12m"])
    y_dev, y_test = dev.y_12m.to_numpy(dtype=int), test.y_12m.to_numpy(dtype=int)

    hazard, at_risk_rows = hazard_fit(fit_all)
    baseline = logistic().fit(features(fit), fit.y_12m)
    trees = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          ExtraTreesClassifier(n_estimators=105, min_samples_leaf=4, max_features=.85,
                                               random_state=SEED, n_jobs=2)).fit(features(fit), fit.y_12m)
    models = {"quarterly_hazard": hazard, "binary_logistic": baseline, "extra_trees": trees}
    cutoffs = {}
    comparison = {}
    results = test[["corp_code", "anchor", "event_date", "followup_end", "y_12m"]].copy()
    for name, model in models.items():
        if name == "quarterly_hazard":
            dev_score = hazard_predict(model, dev)[0][:, -1]
            cumulative, _ = hazard_predict(model, test)
            score = cumulative[:, -1]
            for i in range(4):
                results[f"cumulative_risk_q{i+1}"] = cumulative[:, i]
        else:
            dev_score = model.predict_proba(features(dev))[:, 1]
            score = model.predict_proba(features(test))[:, 1]
        cutoffs[name] = threshold(y_dev, dev_score)
        comparison[name] = dict(development=metrics(y_dev, dev_score, cutoffs[name]),
                                test=metrics(y_test, score, cutoffs[name]))
        results[f"{name}_score"] = score
        results[f"{name}_alert"] = score >= cutoffs[name]
    args.out.mkdir(parents=True, exist_ok=True)
    audit = dict(target="first_observed_OFS_equity_nonpositive", training=inventory(train),
                 fit=inventory(fit_all), development=inventory(dev_all), test=inventory(test_all),
                 at_risk_fit_rows=at_risk_rows, comparison=comparison)
    if args.fresh:
        fresh = normalize_panel(args.fresh, fresh=True)
        fresh = fresh.loc[~fresh.corp_code.isin(set(panel.corp_code) | tr_ids | te_ids)]
        fresh_labels = snapshots(fresh, include_features=False)
        audit["separate_fresh_panel"] = inventory(fresh_labels)
        fresh_labels.to_csv(args.out / "fresh_label_inventory.csv.gz", index=False)
    joblib.dump(dict(target=audit["target"], features=FEATURES, models=models, thresholds=cutoffs),
                args.out / "model.joblib")
    (args.out / "metrics.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    results.to_csv(args.out / "predictions.csv.gz", index=False)
    frame.assign(split=np.where(frame.corp_code.isin(te_ids), "test",
                                np.where(frame.corp_code.isin(groups.index[fit_ids]), "fit", "development"))
                 ).drop(columns=FEATURES).to_csv(args.out / "label_inventory.csv.gz", index=False)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("train", "test", "history", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--fresh", type=Path)
    run(parser.parse_args())
