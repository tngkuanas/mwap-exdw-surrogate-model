
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, logit
from scipy.integrate import cumulative_trapezoid

PARAMS = [
    "Fault Transmissibility",
    "Porosity Multiplier",
    "Permeability Multiplier",
    "Aquifer Pore Volume",
]
CUM_METRICS = ["oil_cum", "gas_cum", "water_cum"]
RATE_METRICS = ["oil_rate", "gas_rate", "water_rate"]
METRICS = CUM_METRICS + RATE_METRICS
YEAR_DAYS = 365.25

METHODS = [
    "Constant",
    "Separate",
    "Liquid-WC-time",
    "Liquid-WC-volume",
]

HEADS = {
    "Constant": [
        "anchor_oil", "anchor_gas", "anchor_water",
        "q_oil", "q_gas", "q_water",
    ],
    "Separate": [
        "anchor_oil", "anchor_gas", "anchor_water",
        "q_oil", "q_water", "gor",
        "D_oil", "water_ratio", "water_k",
    ],
    "Liquid-WC-time": [
        "anchor_oil", "anchor_gas", "anchor_water",
        "q_liquid", "water_cut", "gor", "D_liquid", "beta_time",
    ],
    "Liquid-WC-volume": [
        "anchor_oil", "anchor_gas", "anchor_water",
        "q_liquid", "water_cut", "gor", "D_liquid", "beta_volume",
    ],
}


def load_shared(shared):
    """Load development cases only, plus observed history; audit duplicates."""
    shared = Path(shared)
    unc = pd.read_parquet(shared / "uncertainty_params.parquet")
    unc["case_num"] = pd.to_numeric(unc["case_num"], errors="raise").astype(int)
    unc = unc.sort_values("case_num").set_index("case_num")
    if not unc.index.is_unique:
        raise ValueError("Duplicate uncertainty case numbers.")
    if not set(range(1, 86)).issubset(unc.index):
        raise ValueError("Missing development uncertainty cases.")
    if not unc.loc[1:70, "split"].eq("train").all():
        raise ValueError("Training split mismatch.")
    if not unc.loc[71:85, "split"].eq("validation").all():
        raise ValueError("Validation split mismatch.")
    if not np.isfinite(unc.loc[1:85, PARAMS].to_numpy(float)).all():
        raise ValueError("Nonfinite uncertainty values.")

    prod = pd.read_parquet(
        shared / "production_timeseries.parquet",
        columns=["case_num", "date", "metric", "value"],
    )
    prod["case_num"] = pd.to_numeric(prod["case_num"], errors="coerce")
    prod = prod.loc[(prod["case_num"] >= 0) & (prod["case_num"] <= 85)].copy()
    prod["date"] = pd.to_datetime(prod["date"], errors="coerce")
    prod["value"] = pd.to_numeric(prod["value"], errors="coerce")
    prod = prod.loc[prod["metric"].isin(METRICS)].copy()
    usable = prod["date"].notna() & np.isfinite(prod["value"])
    print("Production rows:", len(prod))
    print("Excluded invalid date/value rows:", int((~usable).sum()))
    prod = prod.loc[usable].copy()

    curves = {}
    for (case, metric), group in prod.groupby(
        ["case_num", "metric"], observed=True
    ):
        stats = group.groupby("date")["value"].agg(["first", "min", "max"])
        if (stats["min"] != stats["max"]).any():
            raise ValueError(f"Conflicting dates: Case {case}, {metric}")
        series = stats["first"].sort_index().astype(float)
        if (series < -1e-8).any():
            raise ValueError(f"Negative values: Case {case}, {metric}")
        if str(metric).endswith("_cum") and (series.diff().dropna() < -1e-6).any():
            raise ValueError(f"Cumulative decreases: Case {case}, {metric}")
        curves[(int(case), str(metric))] = series

    for case in range(1, 86):
        for metric in METRICS:
            if (case, metric) not in curves:
                raise ValueError(f"Missing Case {case}, {metric}")
    return unc, curves


def interpolate_truth(series, dates):
    """Only for labels/evaluation, never for prefix feature construction."""
    dates = pd.DatetimeIndex(dates)
    series = series.dropna().sort_index()
    if len(series) < 2:
        raise ValueError("Insufficient truth observations.")
    if dates.min() < series.index.min() or dates.max() > series.index.max():
        raise ValueError("Requested truth dates exceed native coverage.")
    return np.interp(
        dates.asi8.astype(float),
        series.index.asi8.astype(float),
        series.to_numpy(float),
    )


def prefix_history(curves, case, cutoff):
    """Restrict native samples FIRST; align quarterly afterward."""
    cutoff = pd.Timestamp(cutoff)
    prefixes = {}
    for metric in METRICS:
        series = curves[(int(case), metric)]
        prefixes[metric] = series.loc[series.index <= cutoff].copy()
        if len(prefixes[metric]) < 2:
            raise ValueError(f"Insufficient prefix: {case}, {metric}, {cutoff}")

    start = max(series.index.min() for series in prefixes.values())
    dates = pd.date_range(start.normalize(), cutoff, freq="QS")
    dates = dates[dates >= start]
    dates = dates.union(pd.DatetimeIndex([cutoff])).sort_values()
    if len(dates) < 8:
        raise ValueError("Insufficient quarterly prefix.")

    frame = pd.DataFrame(index=dates)
    lags = {}
    for metric, series in prefixes.items():
        frame[metric] = np.interp(
            dates.asi8.astype(float),
            series.index.asi8.astype(float),
            series.to_numpy(float),
        )
        lags[metric] = (cutoff - series.index[-1]).total_seconds() / 86400

    frame.index.name = "date"
    frame["q_liquid"] = frame["oil_rate"] + frame["water_rate"]
    frame["liquid_cum"] = frame["oil_cum"] + frame["water_cum"]
    frame["water_cut"] = np.divide(
        frame["water_rate"],
        frame["q_liquid"],
        out=np.full(len(frame), np.nan),
        where=frame["q_liquid"].to_numpy() > 1e-8,
    )
    frame["gor"] = np.divide(
        frame["gas_rate"],
        frame["oil_rate"],
        out=np.full(len(frame), np.nan),
        where=frame["oil_rate"].to_numpy() > 1e-8,
    )

    elapsed_years = (
        (frame.index - frame.index[0]).total_seconds().to_numpy()
        / (86400 * YEAR_DAYS)
    )
    for name in ["oil_rate", "water_rate", "q_liquid"]:
        frame[name + "_derivative_per_year"] = np.gradient(
            frame[name].to_numpy(float), elapsed_years
        )
    frame["water_cut_derivative_per_year"] = np.gradient(
        frame["water_cut"].interpolate().bfill().ffill().to_numpy(float),
        elapsed_years,
    )
    return frame, lags


def window_frame(history, window):
    if window == "all":
        result = history.copy()
    else:
        years = int(window[:-1])
        result = history.loc[
            history.index >= history.index[-1] - pd.DateOffset(years=years)
        ].copy()
    result = result.loc[result["oil_rate"] > 1e-8]
    if len(result) < 6:
        raise ValueError(f"Too few producing observations for {window}")
    return result


def anchored_log_decline(t, q, q_cut):
    valid = np.isfinite(q) & (q > 1e-8)
    x = np.asarray(t)[valid]
    y = np.log(np.asarray(q)[valid] / max(float(q_cut), 1e-8))
    denominator = float(np.dot(x, x))
    D = -float(np.dot(x, y)) / max(denominator, 1e-12)
    return float(np.clip(D, 0.0, 3.0))


def target_from_history(history, window="3y"):
    h = window_frame(history, window)
    cutoff = history.index[-1]
    t = (
        (h.index - cutoff).total_seconds().to_numpy()
        / (86400 * YEAR_DAYS)
    )
    last = history.iloc[-1]
    qo = float(last["oil_rate"])
    qw = float(last["water_rate"])
    qg = float(last["gas_rate"])
    ql = qo + qw
    if min(qo, qw, qg) < 0 or ql <= 0:
        raise ValueError("Invalid cutoff rates.")

    fc = float(np.clip(qw / ql, 1e-6, 1 - 1e-6))
    gor_values = h["gor"].to_numpy(float)
    gor = float(np.nanmedian(gor_values))
    if not np.isfinite(gor) or gor <= 0:
        raise ValueError("Cannot estimate produced GOR.")

    D_oil = anchored_log_decline(t, h["oil_rate"].to_numpy(), qo)
    D_liquid = anchored_log_decline(t, h["q_liquid"].to_numpy(), ql)

    water_scale = max(qw, 1.0)
    def water_residual(p):
        ratio, k = p
        predicted = qw * (ratio + (1 - ratio) * np.exp(-k * t))
        return (predicted - h["water_rate"].to_numpy(float)) / water_scale

    water_result = least_squares(
        water_residual,
        x0=[1.2, 0.15],
        bounds=([0.0, 1e-4], [5.0, 3.0]),
        max_nfev=3000,
    )
    if not water_result.success:
        raise RuntimeError(water_result.message)
    water_ratio, water_k = water_result.x

    observed_fc = h["water_cut"].to_numpy(float)
    finite = np.isfinite(observed_fc)
    y = logit(np.clip(observed_fc[finite], 1e-6, 1 - 1e-6)) - logit(fc)

    x_time = t[finite]
    x_volume = (
        h["liquid_cum"].to_numpy(float)[finite] - float(last["liquid_cum"])
    ) / 1e7

    def slope(x, upper):
        value = float(np.dot(x, y)) / max(float(np.dot(x, x)), 1e-12)
        return float(np.clip(value, 0.0, upper))

    beta_time = slope(x_time, 3.0)
    beta_volume = slope(x_volume, 10.0)

    result = {
        "anchor_oil": float(last["oil_cum"]),
        "anchor_gas": float(last["gas_cum"]),
        "anchor_water": float(last["water_cum"]),
        "q_oil": qo, "q_gas": qg, "q_water": qw,
        "q_liquid": ql, "water_cut": fc, "gor": gor,
        "D_oil": D_oil, "D_liquid": D_liquid,
        "water_ratio": float(water_ratio),
        "water_k": float(water_k),
        "water_plateau": float(qw * water_ratio),
        "beta_time": beta_time, "beta_volume": beta_volume,
        "water_fit_rmse": float(
            np.sqrt(np.mean(water_residual(water_result.x) ** 2)) * water_scale
        ),
        "water_ratio_near_bound": bool(water_ratio >= 4.9),
        "water_k_near_bound": bool(water_k <= 1.01e-4 or water_k >= 2.99),
        "D_liquid_at_zero": bool(D_liquid <= 1e-10),
        "fitting_observations": len(h),
    }
    for column in [
        "oil_rate", "water_rate", "q_liquid",
        "oil_rate_derivative_per_year",
        "water_rate_derivative_per_year",
        "q_liquid_derivative_per_year",
        "water_cut_derivative_per_year",
    ]:
        result["late_mean_" + column] = float(h[column].tail(4).mean())
    return result


def fit_trials(history, window="3y", seed=42, n_starts=8):
    """Multistart diagnostic fits: free Arps and unanchored water relaxation."""
    h = window_frame(history, window)
    dates = h.index
    t = (
        (dates - dates[0]).total_seconds().to_numpy()
        / (86400 * YEAR_DAYS)
    )
    qo = h["oil_rate"].to_numpy(float)
    qw = h["water_rate"].to_numpy(float)
    rng = np.random.default_rng(seed)
    rows = []

    def oil_curve(p):
        q0, D, b = p
        if b < 1e-6:
            return q0 * np.exp(-D * t)
        return q0 * (1 + b * D * t) ** (-1 / b)

    def oil_residual(p):
        return np.log(np.maximum(oil_curve(p), 1e-10)) - np.log(np.maximum(qo, 1e-10))

    oil_upper = 10 * max(qo.max(), 1.0)
    water_upper = 5 * max(qw.max(), 1.0)
    water_scale = max(qw.max(), 1.0)

    def water_curve(p):
        q_start, plateau, k = p
        return plateau + (q_start - plateau) * np.exp(-k * t)

    def water_residual(p):
        return (water_curve(p) - qw) / water_scale

    for start in range(n_starts):
        oil_fit = least_squares(
            oil_residual,
            x0=[
                min(qo[0] * rng.uniform(0.8, 1.2), 0.9 * oil_upper),
                rng.uniform(0.01, 0.5),
                rng.uniform(0.01, 0.99),
            ],
            bounds=([1e-6, 1e-5, 0.0], [oil_upper, 3.0, 1.0]),
            max_nfev=3000,
        )
        water_fit = least_squares(
            water_residual,
            x0=[
                max(qw[0], 1e-6),
                rng.uniform(0.1, 0.9) * water_upper,
                rng.uniform(0.01, 0.8),
            ],
            bounds=([0.0, 0.0, 1e-5], [water_upper, water_upper, 3.0]),
            max_nfev=3000,
        )

        for kind, fit, curve, actual, scale in [
            ("Arps", oil_fit, oil_curve, qo, max(qo.max(), 1.0)),
            ("Water", water_fit, water_curve, qw, water_scale),
        ]:
            singular = np.linalg.svd(fit.jac, compute_uv=False)
            condition = float(singular[0] / max(singular[-1], 1e-15))
            row = {
                "kind": kind,
                "start_number": start,
                "success": bool(fit.success),
                "cost": float(fit.cost),
                "history_NRMSE": float(
                    np.sqrt(np.mean((curve(fit.x) - actual) ** 2)) / scale
                ),
                "jacobian_condition_number": condition,
                "p0": float(fit.x[0]),
                "p1": float(fit.x[1]),
                "p2": float(fit.x[2]),
                "plateau_upper": water_upper if kind == "Water" else np.nan,
            }
            rows.append(row)
    return rows


def encode_targets(frame, method):
    columns = HEADS[method]
    z = frame[columns].to_numpy(float).copy()
    for j, column in enumerate(columns):
        if column == "water_cut":
            z[:, j] = logit(np.clip(z[:, j], 1e-6, 1 - 1e-6))
        else:
            z[:, j] = np.log(np.maximum(z[:, j], 1e-8))
    return z


def decode_targets(z, method):
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z[None, :]
    records = []
    for values in z:
        record = {}
        for column, value in zip(HEADS[method], values):
            if column == "water_cut":
                record[column] = float(np.clip(expit(value), 1e-6, 1 - 1e-6))
            else:
                record[column] = float(np.exp(np.clip(value, -18, 30)))
        for column in ["D_oil", "D_liquid", "water_k", "beta_time"]:
            if column in record:
                record[column] = float(np.clip(record[column], 0.0, 3.0))
        if "water_ratio" in record:
            record["water_ratio"] = float(np.clip(record["water_ratio"], 0, 5))
        if "beta_volume" in record:
            record["beta_volume"] = float(np.clip(record["beta_volume"], 0, 10))
        records.append(record)
    return records


def forecast(record, method, cutoff, dates):
    cutoff = pd.Timestamp(cutoff)
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0 or dates.min() <= cutoff:
        raise ValueError("Forecast dates must be strictly after cutoff.")
    dense = pd.date_range(cutoff, dates.max(), freq="D")
    dense = dense.union(dates).sort_values()
    elapsed_days = (dense - cutoff).total_seconds().to_numpy() / 86400
    t = elapsed_days / YEAR_DAYS

    if method == "Constant":
        rates = np.tile(
            [record["q_oil"], record["q_gas"], record["q_water"]],
            (len(dense), 1),
        )
    elif method == "Separate":
        qo = record["q_oil"] * np.exp(-record["D_oil"] * t)
        qw = record["q_water"] * (
            record["water_ratio"]
            + (1 - record["water_ratio"]) * np.exp(-record["water_k"] * t)
        )
        rates = np.column_stack([qo, record["gor"] * qo, qw])
    elif method in ["Liquid-WC-time", "Liquid-WC-volume"]:
        ql = record["q_liquid"] * np.exp(-record["D_liquid"] * t)
        if method == "Liquid-WC-time":
            progress = record["beta_time"] * t
        else:
            liquid_increment = cumulative_trapezoid(
                ql, elapsed_days, initial=0
            )
            progress = record["beta_volume"] * liquid_increment / 1e7
        fc = expit(
            logit(np.clip(record["water_cut"], 1e-6, 1 - 1e-6)) + progress
        )
        qo = ql * (1 - fc)
        qw = ql * fc
        rates = np.column_stack([qo, record["gor"] * qo, qw])
    else:
        raise ValueError(method)

    if not np.isfinite(rates).all() or (rates < -1e-8).any():
        raise ValueError("Invalid forecast rates.")
    anchor = np.array([
        record["anchor_oil"], record["anchor_gas"], record["anchor_water"]
    ])
    cumulative = anchor + cumulative_trapezoid(
        rates, elapsed_days, axis=0, initial=0
    )
    positions = dense.get_indexer(dates)
    return cumulative[positions], rates[positions]
