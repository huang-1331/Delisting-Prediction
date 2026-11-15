"""Unchanged 12-month feature transformation from the frozen first-stage experiment."""
import numpy as np
import pandas as pd

SEED = 38271

BASE = ['impairment_ratio','impairment_delta_1q','impairment_mean_4q','equity_delta_4q_assets','debt_assets','debt_assets_delta_1q','current_ratio','working_capital_assets','roa_ttm','operating_roa_ttm','loss_fraction_4q','retained_assets','retained_deficit_to_capital','erosion_pressure_at_label_limit','projected_impairment','liability_to_capital','loss_to_capital_proxy']

def matrices(frame):
    base = frame[BASE].astype(float).replace([np.inf, -np.inf], np.nan)
    x = base.copy()
    cap_asset = x.debt_assets / x.liability_to_capital.where(x.liability_to_capital > 0)
    equity = 1 - x.debt_assets
    margin = (.35 - x.impairment_ratio).clip(lower=.01)
    x['equity_assets'] = equity
    x['paid_cap_assets'] = cap_asset
    x['net_loss_cover'] = -x.roa_ttm / equity.where(equity > 0)
    x['operating_loss_cover'] = -x.operating_roa_ttm / equity.where(equity > 0)
    x['annual_capital_cover'] = x.loss_to_capital_proxy / margin
    x['equity_drawdown'] = -x.equity_delta_4q_assets / (equity - x.equity_delta_4q_assets).where(lambda s:s>0)
    x['nonoperating_gap'] = x.operating_roa_ttm - x.roa_ttm
    x['loss_on_working_cap'] = -x.roa_ttm / x.working_capital_assets.where(x.working_capital_assets > 0)
    x['accrual_pressure'] = x.retained_deficit_to_capital / (1-x.impairment_ratio).where(x.impairment_ratio<1)
    x['debt_accel'] = x.debt_assets_delta_1q - (-x.equity_delta_4q_assets / 4)
    x['impairment_slope'] = x.impairment_ratio - x.impairment_mean_4q
    for months in (12,24):
        x[f'projection_{months}'] = x.impairment_ratio + (months/12 + .5) * x.loss_to_capital_proxy
        x[f'projection_trend_{months}'] = x.impairment_ratio + (months/3 + 2) * x.impairment_delta_1q
    sort = x.assign(company_key=frame.corp_code, quarter=pd.PeriodIndex(frame.feature_end, freq='Q').asi8).sort_values(['company_key','quarter'])
    group = sort.groupby('company_key', sort=False)
    gap = group.quarter.diff()
    history = {}
    for key in ('impairment_ratio','debt_assets','loss_to_capital_proxy','roa_ttm','operating_roa_ttm','working_capital_assets'):
        past = group[key].shift()
        delta = (sort[key]-past).where(gap.between(1,4))/gap
        history[f'{key}_slope'] = delta
        history[f'{key}_accel'] = (delta-delta.groupby(sort.company_key).shift()).where(gap.between(1,4))
        history[f'{key}_change'] = (sort[key]-group[key].shift(2)).where((sort.quarter-group.quarter.shift(2)).between(2,6))
        history[f'{key}_relative'] = delta/(past.abs()+.01)
    base_fin = ['impairment_ratio','debt_assets','working_capital_assets','roa_ttm','operating_roa_ttm','retained_assets','loss_to_capital_proxy','current_ratio','loss_fraction_4q','equity_delta_4q_assets','paid_cap_assets','net_loss_cover','annual_capital_cover','impairment_slope','debt_assets_delta_1q','impairment_delta_1q']
    return {'base':base,'structural':x[base_fin].replace([np.inf,-np.inf],np.nan),'trajectory':x.join(pd.DataFrame(history)).replace([np.inf,-np.inf],np.nan)}
