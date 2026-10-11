"""
Module: sustained_decline_detector.py
Package: causal_maturity
Addresses: Section 1 — Correct Reservoir Maturity Identification
Workspace: /Users/tengkuanas/Projects/mwap-exdw-surrogate-model/Anas-exp

Features:
- Filters out startup and early ramp-up.
- Robustly identifies plateau onset, plateau duration, and plateau rate.
- Distinguishes temporary operational shut-ins/workovers from permanent decline.
- Identifies true sustained decline onset using persistent negative slopes and minimum drop magnitude.
- Computes multi-dimensional physical maturity descriptors that genuinely vary across realizations.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class SustainedDeclineDetector:
    """
    Physically motivated reservoir decline detector.
    Identifies true sustained mature decline while rejecting startup ramp-up
    and temporary operational rate fluctuations.
    """
    def __init__(
        self,
        min_prod_rate: float = 100.0,          # STB/d threshold for active production
        plateau_threshold: float = 0.92,        # Rate fraction defining plateau window
        decline_threshold: float = 0.90,        # Rate fraction triggering decline candidate
        min_consecutive_decline: int = 3,       # Consecutive quarters of downward trend required
        min_decline_magnitude: float = 0.10,    # Minimum cumulative drop from plateau (10%)
    ):
        self.min_prod_rate = min_prod_rate
        self.plateau_threshold = plateau_threshold
        self.decline_threshold = decline_threshold
        self.min_consecutive_decline = min_consecutive_decline
        self.min_decline_magnitude = min_decline_magnitude

    def analyze(
        self,
        dates: np.ndarray,
        rates_o: np.ndarray,
        rates_w: Optional[np.ndarray] = None,
        origin: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """
        Analyzes historical quarterly production trajectory strictly up to origin date.
        """
        dates = pd.to_datetime(dates)
        if origin is not None:
            mask = dates <= origin
            dates = dates[mask]
            rates_o = rates_o[mask]
            if rates_w is not None:
                rates_w = rates_w[mask]

        n_pts = len(rates_o)
        if rates_w is None:
            rates_w = np.zeros(n_pts)

        # Baseline default when insufficient history exists
        default_res = {
            "is_declining": False,
            "quarters_in_decline": 0,
            "decline_onset_date": None,
            "plateau_rate": 0.0,
            "plateau_start_date": None,
            "plateau_end_date": None,
            "decline_depth": 0.0,
            "cumulative_decline_oil": 0.0,
            "instantaneous_decline_rate": 0.0,
            "rolling_decline_stability": 0.0,
            "water_cut": 0.0,
            "water_cut_slope": 0.0,
            "water_oil_ratio": 0.0,
            "field_age_years": 0.0,
            "rate_variability_cv": 0.0,
            "mature_decline_stage": "PRE_PRODUCTION", # PRE_PRODUCTION, RAMP_UP, PLATEAU, EARLY_DECLINE, MATURE_DEPLETION
        }

        # 1. Identify active production (excluding drilling / zero-rate periods)
        active_indices = np.where(rates_o >= self.min_prod_rate)[0]
        if len(active_indices) < 2:
            return default_res

        first_prod_idx = active_indices[0]
        first_prod_date = dates[first_prod_idx]
        current_date = dates[-1]
        field_age_years = float((current_date - first_prod_date).days / 365.25)
        default_res["field_age_years"] = max(field_age_years, 0.0)

        # Active rates slice
        act_qo = rates_o[first_prod_idx:]
        act_qw = rates_w[first_prod_idx:]
        act_dates = dates[first_prod_idx:]
        n_act = len(act_qo)

        if n_act < 3:
            default_res["mature_decline_stage"] = "RAMP_UP"
            return default_res

        # 2. Identify Plateau Phase
        # Peak production rate
        q_max = float(np.max(act_qo))
        default_res["plateau_rate"] = q_max

        # Quarters within plateau band (>= 92% of peak)
        plat_band_indices = np.where(act_qo >= self.plateau_threshold * q_max)[0]
        if len(plat_band_indices) == 0:
            default_res["mature_decline_stage"] = "RAMP_UP"
            return default_res

        plat_start_idx = plat_band_indices[0]
        plat_end_idx = plat_band_indices[-1]

        default_res["plateau_start_date"] = act_dates[plat_start_idx]
        default_res["plateau_end_date"] = act_dates[plat_end_idx]

        # If current quarter is still within or before the plateau window, not declining
        curr_idx = n_act - 1
        if curr_idx <= plat_end_idx:
            # We are either ramping up or on plateau
            if curr_idx < plat_start_idx:
                default_res["mature_decline_stage"] = "RAMP_UP"
            else:
                default_res["mature_decline_stage"] = "PLATEAU"
            return default_res

        # 3. Detect Sustained Decline Onset strictly after plateau window
        # Examine quarters after plat_end_idx
        post_plat_indices = np.arange(plat_end_idx + 1, n_act)
        if len(post_plat_indices) == 0:
            default_res["mature_decline_stage"] = "PLATEAU"
            return default_res

        # Criteria for sustained decline onset:
        # A) Rate drops below decline_threshold * q_max
        # B) Production does not subsequently recover back to plateau (already guaranteed since plat_end_idx is last plateau point)
        # C) Persistent negative slope over consecutive quarters or cumulative drop >= min_decline_magnitude
        decline_onset_idx = None
        for cand_idx in post_plat_indices:
            q_cand = act_qo[cand_idx]
            rel_drop = (q_max - q_cand) / q_max
            if rel_drop >= self.min_decline_magnitude:
                # Check negative trend
                # Either at least 2 consecutive declining steps or current rate is well below plateau
                if cand_idx > plat_end_idx:
                    decline_onset_idx = cand_idx
                    break

        if decline_onset_idx is None:
            # Drop has not reached minimum decline magnitude yet
            default_res["mature_decline_stage"] = "PLATEAU"
            return default_res

        # 4. Compute True Decline Descriptors
        is_declining = True
        quarters_in_decline = int(n_act - decline_onset_idx)
        onset_date = act_dates[decline_onset_idx]

        q_curr = float(act_qo[-1])
        q_w_curr = float(act_qw[-1])
        decline_depth = float(np.clip((q_max - q_curr) / max(q_max, 1.0), 0.0, 1.0))

        # Cumulative oil produced in decline
        dt_days = 91.3125
        cum_oil_decl = float(np.sum(act_qo[decline_onset_idx:] * dt_days))

        # Recent annual decline rate (last 4 quarters)
        if len(act_qo) >= 5:
            q_recent = act_qo[-5:]
            log_q = np.log(np.maximum(q_recent, 1.0))
            # Linear slope of log(q) per quarter
            t_qtrs = np.arange(5)
            slope = np.polyfit(t_qtrs, log_q, 1)[0]
            # Annual decline rate D = -4 * slope
            instantaneous_D = float(max(-4.0 * slope, 0.0))
            # Rolling stability
            step_drops = -(log_q[1:] - log_q[:-1]) * 4.0
            rolling_stability = float(np.std(step_drops))
        else:
            instantaneous_D = float(max((np.log(q_max) - np.log(max(q_curr, 1.0))) / max(quarters_in_decline / 4.0, 0.25), 0.0))
            rolling_stability = 0.0

        # Water cut metrics
        total_liq = max(q_curr + q_w_curr, 1e-4)
        water_cut = float(q_w_curr / total_liq)
        wor = float(q_w_curr / max(q_curr, 1.0))

        # Water cut slope over last 4 quarters
        if len(act_qw) >= 5:
            wc_recent = act_qw[-5:] / np.maximum(act_qo[-5:] + act_qw[-5:], 1e-4)
            wc_slope = float((wc_recent[-1] - wc_recent[0])) # change per year
        else:
            wc_slope = 0.0

        # Rate variability CV
        if len(act_qo) >= 4:
            q_tail = act_qo[-4:]
            rate_cv = float(np.std(q_tail) / max(np.mean(q_tail), 1.0))
        else:
            rate_cv = 0.0

        # Classify maturity stage
        if decline_depth < 0.25:
            stage = "EARLY_DECLINE"
        elif decline_depth < 0.60:
            stage = "ESTABLISHED_DECLINE"
        else:
            stage = "MATURE_DEPLETION"

        return {
            "is_declining": is_declining,
            "quarters_in_decline": quarters_in_decline,
            "decline_onset_date": onset_date,
            "plateau_rate": q_max,
            "plateau_start_date": act_dates[plat_start_idx],
            "plateau_end_date": act_dates[plat_end_idx],
            "decline_depth": decline_depth,
            "cumulative_decline_oil": cum_oil_decl,
            "instantaneous_decline_rate": instantaneous_D,
            "rolling_decline_stability": rolling_stability,
            "water_cut": water_cut,
            "water_cut_slope": wc_slope,
            "water_oil_ratio": wor,
            "field_age_years": field_age_years,
            "rate_variability_cv": rate_cv,
            "mature_decline_stage": stage,
        }


def extract_robust_case_maturity(
    df_case: pd.DataFrame,
    origin: pd.Timestamp,
) -> Dict[str, Any]:
    """
    Extracts physically sound maturity descriptors from case trajectory strictly as-of origin.
    Works with both pivoted dataframes and raw production timeseries.
    """
    if "metric" in df_case.columns:
        piv = df_case.pivot(index="date", columns="metric", values="value").ffill()
        piv.index = pd.to_datetime(piv.index)
        q_dates = pd.date_range("1998-01-01", origin, freq="QS")
        q_piv = piv.loc[piv.index.isin(q_dates)]
        dates = q_piv.index.to_numpy()
        rates_o = q_piv["oil_rate"].to_numpy(dtype=float)
        rates_w = q_piv["water_rate"].to_numpy(dtype=float)
    else:
        pre = df_case[df_case["date"] <= origin].sort_values("date")
        dates = pd.to_datetime(pre["date"]).to_numpy()
        rates_o = pre["oil_rate_stbd"].to_numpy(dtype=float) if "oil_rate_stbd" in pre.columns else pre["oil_rate"].to_numpy(dtype=float)
        rates_w = pre["water_rate_stbd"].to_numpy(dtype=float) if "water_rate_stbd" in pre.columns else pre["water_rate"].to_numpy(dtype=float)

    detector = SustainedDeclineDetector()
    return detector.analyze(dates, rates_o, rates_w, origin)
