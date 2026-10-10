"""
Family 1: Parametric Decline Models.
Implements separate-phase exponential decline with strict b=0.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd

YEAR_DAYS = 365.25

def fit_exponential_decline(dates: pd.DatetimeIndex, rates: np.ndarray, origin: pd.Timestamp) -> Tuple[float, float]:
    """
    Fits strict b=0 exponential decline: ln(q(t)) = ln(q_0) - D * t (in years)
    Returns (q_0, D) with non-negative D.
    """
    valid = np.isfinite(rates) & (rates > 0)
    if valid.sum() < 2:
        q_last = float(rates[-1]) if len(rates) > 0 and np.isfinite(rates[-1]) and rates[-1] > 0 else 1.0
        return q_last, 0.0
    
    t_years = (dates[valid] - origin).total_seconds() / (86400.0 * YEAR_DAYS)
    log_q = np.log(rates[valid])
    
    # Linear regression: log_q = a - D * t
    if np.ptp(t_years) > 1e-4:
        poly = np.polyfit(t_years, log_q, deg=1)
        D = -poly[0]
        q_0 = np.exp(poly[1])
    else:
        D = 0.0
        q_0 = np.exp(np.mean(log_q))
        
    D = max(0.0, float(D))
    q_0 = max(1e-4, float(q_0))
    return q_0, D

class StrictExponentialModel:
    """
    Separate-phase strict b=0 exponential decline model.
    Oil: exponential decline fitted over window_years.
    Gas: historical median GOR * oil rate.
    Water: exponential decline or terminal rate.
    """
    def __init__(self, window_years: float = 3.0):
        self.window_years = window_years
        self.model_id = "StrictExponential_b0"
        
    def fit_predict_case(
        self,
        case: int,
        origin: pd.Timestamp,
        forecast_dates: pd.DatetimeIndex,
        curves: Dict[Tuple[int, str], pd.Series],
        horizon_years: int = 3,
    ) -> pd.DataFrame:
        origin = pd.Timestamp(origin)
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        
        # 1. Historical windows
        oil_s = curves[(case, "oil_rate")].loc[:origin]
        gas_s = curves[(case, "gas_rate")].loc[:origin]
        water_s = curves[(case, "water_rate")].loc[:origin]
        
        oil_cum_s = curves[(case, "oil_cum")].loc[:origin]
        gas_cum_s = curves[(case, "gas_cum")].loc[:origin]
        water_cum_s = curves[(case, "water_cum")].loc[:origin]
        
        anchor_oil = float(oil_cum_s.iloc[-1])
        anchor_gas = float(gas_cum_s.iloc[-1])
        anchor_water = float(water_cum_s.iloc[-1])
        
        start_fit = origin - pd.DateOffset(years=int(self.window_years))
        oil_fit = oil_s.loc[start_fit:origin]
        gas_fit = gas_s.loc[start_fit:origin]
        water_fit = water_s.loc[start_fit:origin]
        
        # Fit Oil
        q_oil_0, D_oil = fit_exponential_decline(oil_fit.index, oil_fit.to_numpy(float), origin)
        # Use terminal observed rate as q_0 anchor for rate continuity
        if len(oil_fit) > 0 and oil_fit.iloc[-1] > 0:
            q_oil_0 = float(oil_fit.iloc[-1])
            
        # Fit GOR (scf/stb)
        with np.errstate(divide='ignore', invalid='ignore'):
            gor_series = gas_fit.to_numpy(float) / oil_fit.to_numpy(float)
            valid_gor = gor_series[np.isfinite(gor_series) & (gor_series > 0)]
            median_gor = float(np.median(valid_gor)) if len(valid_gor) > 0 else 1.0
            
        # Fit Water
        q_water_0, D_water = fit_exponential_decline(water_fit.index, water_fit.to_numpy(float), origin)
        if len(water_fit) > 0 and water_fit.iloc[-1] > 0:
            q_water_0 = float(water_fit.iloc[-1])
            
        # Forecast increments
        dt_years = (forecast_dates - origin).total_seconds() / (86400.0 * YEAR_DAYS)
        
        # Cumulative increment formulas: Q = q0 * (1 - exp(-D*t)) / D
        if D_oil > 1e-6:
            cum_oil_inc = q_oil_0 * YEAR_DAYS * (1.0 - np.exp(-D_oil * dt_years)) / D_oil
        else:
            cum_oil_inc = q_oil_0 * YEAR_DAYS * dt_years
            
        cum_gas_inc = median_gor * cum_oil_inc
        
        if D_water > 1e-6:
            cum_water_inc = q_water_0 * YEAR_DAYS * (1.0 - np.exp(-D_water * dt_years)) / D_water
        else:
            cum_water_inc = q_water_0 * YEAR_DAYS * dt_years
            
        oil_cum_pred = anchor_oil + cum_oil_inc
        gas_cum_pred = anchor_gas + cum_gas_inc
        water_cum_pred = anchor_water + cum_water_inc
        
        rows = []
        for i, d in enumerate(forecast_dates):
            rows.append({"model_id": self.model_id, "case_num": case, "origin": origin, "cutoff": origin, "horizon_years": int(horizon_years), "date": d, "phase": "oil_cum", "prediction": float(oil_cum_pred[i])})
            rows.append({"model_id": self.model_id, "case_num": case, "origin": origin, "cutoff": origin, "horizon_years": int(horizon_years), "date": d, "phase": "gas_cum", "prediction": float(gas_cum_pred[i])})
            rows.append({"model_id": self.model_id, "case_num": case, "origin": origin, "cutoff": origin, "horizon_years": int(horizon_years), "date": d, "phase": "water_cum", "prediction": float(water_cum_pred[i])})
            
        return pd.DataFrame(rows)
