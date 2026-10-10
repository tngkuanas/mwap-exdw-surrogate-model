"""
hz_model.py — Geology-feature regressors + hyperbolic decline (Hazeem's track).

Idea
----
Arps hyperbolic decline is  dq/dt = -D q,  with  D = K * q^b.
In logs:  ln D = ln K + b ln q   -> a LINEAR model, i.e. exactly what Ridge fits.
We let ln K depend on the reservoir's state at the cutoff (recovery factor, water cut)
and on geology (the engineered grid features or the raw 4 parameters).

    1. Build a quarterly panel per case (rates, cumulatives, water cut, recovery factor).
    2. Label each quarter c with the realised forward decline over the next year:
           D_fwd(c) = -ln(q(c+1y)/q(c))          (oil)
           B_fwd(c) = logit(wc(c+1y)) - logit(wc(c))   (water-cut growth)
       Only quarters with c + 1y <= cutoff T are used for training (no time leakage).
    3. Ridge / SVR:  ln D_fwd ~ [ln q, state, geology],  B_fwd ~ [same].
    4. Forecast from T:  oil = hyperbolic with D_T = model(state_T), b from the panel;
       water from the predicted water-cut growth; gas = GOR_T * oil; integrate from
       the true cumulative anchor at T.

Scoring uses Anas's BacktestEngine (increment NRMSE, experiments 2003/2004/2005).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.special import expit, logit
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

YEAR = 365.25
QDATES = pd.date_range("1998-01-01", "2008-01-01", freq="QS")
EXPERIMENTS = [("2003-01-01", 3), ("2003-01-01", 5), ("2004-01-01", 3), ("2005-01-01", 3)]
PLATEAU = 15400.0   # oil rate on the initial plateau is 15,508 STB/d


# ── 1. quarterly panel ────────────────────────────────────────────
def build_panel(curves, cases, geo):
    rows = []
    for c in cases:
        get = lambda m: np.interp(QDATES.asi8, curves[(c, m)].index.asi8,
                                  curves[(c, m)].to_numpy(float))
        qo, qw, qg = get("oil_rate"), get("water_rate"), get("gas_rate")
        no, nw, ng = get("oil_cum"), get("water_cum"), get("gas_cum")
        d = pd.DataFrame({"date": QDATES, "case_num": c, "qo": qo, "qw": qw, "qg": qg,
                          "no": no, "nw": nw, "ng": ng})
        rows.append(d)
    p = pd.concat(rows, ignore_index=True)
    p["t"] = (p["date"] - QDATES[0]).dt.days / YEAR
    p["wc"] = np.clip(p.qw / np.maximum(p.qo + p.qw, 1e-9), 1e-4, 1 - 1e-4)
    p["gor"] = p.qg / np.maximum(p.qo, 1e-9)
    p = p.join(geo[["OOIP_total", "PV_aquifer"]], on="case_num")
    # STB vs reservoir bbl: ratio is a constant per case, fine as a scaled index
    p["rf"] = p["no"] / p["OOIP_total"]
    p["wf"] = p["nw"] / p["PV_aquifer"]
    p["ln_qo"] = np.log(np.maximum(p.qo, 1.0))
    p["logit_wc"] = logit(p.wc)
    g = p.groupby("case_num")
    # backward-looking recent decline (uses past only) — a state variable
    p["D_recent"] = -(p["ln_qo"] - g["ln_qo"].shift(4))
    p["B_recent"] = p["logit_wc"] - g["logit_wc"].shift(4)
    # forward labels (one year ahead)
    p["D_fwd"] = -(g["ln_qo"].shift(-4) - p["ln_qo"])
    p["B_fwd"] = g["logit_wc"].shift(-4) - p["logit_wc"]
    p["date_fwd"] = g["date"].shift(-4)
    # water-rate growth (ln qw change per year), forward label + recent state
    p["ln_qw"] = np.log(np.maximum(p.qw, 1.0))
    p["E_recent"] = p["ln_qw"] - g["ln_qw"].shift(4)
    p["E_fwd"] = g["ln_qw"].shift(-4) - p["ln_qw"]
    # grid-free versions of the cumulative state (no OOIP / aquifer normalisation)
    p["no_s"] = p["no"] / 1e7
    p["nw_s"] = p["nw"] / 1e7
    return p


# ── 2. feature sets ───────────────────────────────────────────────
STATE = ["ln_qo", "rf", "logit_wc", "D_recent", "B_recent", "wf"]          # rf, wf use grid OOIP / aquifer PV
STATE_NOGRID = ["ln_qo", "no_s", "logit_wc", "D_recent", "B_recent", "nw_s"]  # same, no grid information
RAW = ["Fault Transmissibility", "Porosity Multiplier", "Permeability Multiplier",
       "Aquifer Pore Volume"]
GEO = ["OOIP_total", "OOIP_tight", "T_tight", "T_cross_fault", "PV_aquifer",
       "aquifer_to_oil_ratio", "tight_drainage_time", "west_connectivity",
       "diffusivity_proxy"]
FEATURE_SETS = {
    "production state only (no grid)": STATE_NOGRID,
    "production state + raw 4 params": STATE_NOGRID + RAW,
    "state normalised by grid (rf, wf)": STATE,
    "grid-normalised state + raw 4 params": STATE + RAW,
    "grid-normalised state + geology features": STATE + GEO,
}


def design(panel, geo, cols):
    X = panel[[c for c in cols if c in panel.columns]].copy()
    for c in cols:
        if c not in X.columns:
            X[c] = panel["case_num"].map(geo[c]).to_numpy()
    # geology/params enter in logs (multiplicative physics); state as is
    for c in cols:
        if c in RAW or c in GEO:
            X[c] = np.log(np.abs(X[c].astype(float)) + 1e-30)
    return X.to_numpy(float)


def make_model(kind):
    if kind == "Ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 30)))
    if kind == "SVR":
        return make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.01, gamma="scale"))
    raise ValueError(kind)


# ── 3. training rows for a cutoff (no time leakage) ───────────────
def train_rows(panel, cases, cutoff):
    m = (panel.case_num.isin(cases) & (panel.date_fwd <= cutoff) &
         panel.D_recent.notna() & panel.D_fwd.notna() & (panel.qo > 1))
    return panel.loc[m]


def estimate_b(panel, cases, cutoff, min_rows=60):
    """Arps b = within-case slope of ln D on ln q, using only rows AFTER each case's
    peak decline (the plateau-to-decline transition would bias b towards 0).
    Falls back to b = 0 (exponential) when too few post-peak rows exist before cutoff."""
    r = train_rows(panel, cases, cutoff)
    r = r[r.D_fwd > 1e-3]
    if len(r) == 0:
        return 0.0
    peak = r.loc[r.groupby("case_num").D_fwd.idxmax(), ["case_num", "date"]].set_index("case_num").date
    r = r[r.date >= r.case_num.map(peak)]
    # need at least 3 post-peak points per case to define a slope
    r = r[r.groupby("case_num").date.transform("size") >= 3]
    if len(r) < min_rows:
        return 0.0
    x = r.ln_qo - r.groupby("case_num").ln_qo.transform("mean")
    y = np.log(r.D_fwd) - np.log(r.D_fwd).groupby(r.case_num).transform("mean")
    b = float((x * y).sum() / max((x * x).sum(), 1e-12))
    return float(np.clip(b, 0.0, 1.0))


# ── 4. forecast decoder ───────────────────────────────────────────
def hyperbolic_forecast(state, D0, b, B, dates, cutoff, water=None):
    """state: panel row at cutoff. D0 = decline at cutoff (1/yr), b = Arps b,
    B = water-cut logit growth per year. Returns cumulative oil, gas, water."""
    cutoff = pd.Timestamp(cutoff)
    dense = pd.date_range(cutoff, max(dates), freq="D").union(dates)
    days = (dense - cutoff).days.to_numpy(float)
    t = days / YEAR
    D0 = float(np.clip(D0, 0.0, 2.0))
    if b < 1e-4:
        qo = state.qo * np.exp(-D0 * t)
    else:
        qo = state.qo * (1 + b * D0 * t) ** (-1 / b)
    if water is None:      # learned water-cut growth
        wc = expit(state.logit_wc + max(B, 0.0) * t)
        qw = qo * wc / (1 - wc)
    else:                  # history-fitted water relaxation (Anas's 'Separate' water)
        qw = state.qw * (water["water_ratio"] + (1 - water["water_ratio"]) * np.exp(-water["water_k"] * t))
    qg = state.gor * qo
    rates = np.column_stack([qo, qg, qw])
    cum = np.array([state["no"], state["ng"], state["nw"]]) + cumulative_trapezoid(
        rates, days, axis=0, initial=0)
    pos = dense.get_indexer(dates)
    return cum[pos]


def experiment_dates(origin, horizon):
    o = pd.Timestamp(origin)
    return pd.date_range(o + pd.offsets.QuarterBegin(startingMonth=1),
                         o + pd.DateOffset(years=horizon), freq="QS")


# ── 5. run one configuration over all experiments ─────────────────
def run(panel, geo, feature_cols, kind, model_id, dev_cases=range(1, 86),
        train_cases=range(1, 71), seed=42, target="lin", water_fits=None, guard=None):
    """target: 'lin' fits D_fwd, 'log' fits ln D_fwd, 'resid' fits D_fwd - D_recent
    (falls back to the recently observed decline when extrapolating).
    water_fits: dict (case, cutoff) -> {'water_ratio','water_k'} to use history water."""
    train_cases = list(train_cases)
    folds = {}
    for f, (_, hold) in enumerate(KFold(5, shuffle=True, random_state=seed).split(train_cases)):
        for i in hold:
            folds[train_cases[i]] = f
    out = []
    for origin, horizon in EXPERIMENTS:
        cutoff = pd.Timestamp(origin)
        dates = experiment_dates(origin, horizon)
        groups = {f: [c for c in train_cases if folds[c] == f] for f in range(5)}
        groups["val"] = []
        for key, hold in groups.items():
            fit_cases = [c for c in train_cases if c not in hold]
            score_cases = hold if key != "val" else [c for c in dev_cases if c > 70]
            if not score_cases:
                continue
            tr = train_rows(panel, fit_cases, cutoff)
            X = design(tr, geo, feature_cols)
            if target == "log":
                yD = np.log(np.clip(tr.D_fwd, 1e-4, None))
            elif target == "resid":
                yD = tr.D_fwd - tr.D_recent
            else:
                yD = tr.D_fwd.clip(lower=0)
            mD = make_model(kind).fit(X, yD)
            mB = make_model(kind).fit(X, tr.B_fwd)
            b = estimate_b(panel, fit_cases, cutoff)
            st = panel[(panel.date == cutoff) & panel.case_num.isin(score_cases)]
            Xs = design(st, geo, feature_cols)
            if target == "log":
                D0 = np.exp(mD.predict(Xs))
            elif target == "resid":
                D0 = np.clip(st.D_recent.to_numpy() + mD.predict(Xs), 0, None)
            else:
                D0 = np.clip(mD.predict(Xs), 0, None)
            if guard is not None:   # keep D0 within [lo, hi] x recently observed decline
                rec = st.D_recent.to_numpy()
                D0 = np.clip(D0, guard[0] * rec, guard[1] * rec)
            B = mB.predict(Xs)
            for (_, s), d0, bb in zip(st.iterrows(), D0, B):
                w = None if water_fits is None else water_fits[(int(s.case_num), cutoff)]
                cum = hyperbolic_forecast(s, d0, b, bb, dates, cutoff, water=w)
                for j, ph in enumerate(["oil_cum", "gas_cum", "water_cum"]):
                    out.append(pd.DataFrame({
                        "model_id": model_id, "case_num": int(s.case_num),
                        "origin": cutoff, "horizon_years": horizon,
                        "date": dates, "phase": ph, "prediction": cum[:, j],
                        "b_used": b}))
    return pd.concat(out, ignore_index=True)


# ── 6. roll-forward decoder (state re-evaluated every quarter, all cases at once) ──
def rollforward_batch(hists, geo, cols, mD, mE, dates, oil="roll", D0=None, b=0.0):
    """hists: list of per-case panel frames up to and including the cutoff.
    oil='roll' -> oil decline re-predicted each quarter; 'hyperbolic' -> fixed D0[i], b.
    Returns array (n_cases, n_dates, 3) of cumulative oil, gas, water."""
    n = len(hists)
    cases = np.array([int(h.case_num.iloc[0]) for h in hists])
    tail = lambda k, m: np.array([h[k].to_numpy(float)[-m] for h in hists])
    qo_hist = [tail("qo", m) for m in (5, 4, 3, 2, 1)]       # last 5 quarters
    qw_hist = [tail("qw", m) for m in (5, 4, 3, 2, 1)]
    no, nw, ng = tail("no", 1), tail("nw", 1), tail("ng", 1)
    gor = tail("gor", 1)
    ooip = geo.loc[cases, "OOIP_total"].to_numpy(); pva = geo.loc[cases, "PV_aquifer"].to_numpy()
    cutoff = hists[0].date.iloc[-1]
    steps = pd.date_range(cutoff, max(dates), freq="QS")
    res = {}
    lg = lambda x: np.log(np.maximum(x, 1.0))
    wcf = lambda o, w: np.clip(w / np.maximum(o + w, 1e-9), 1e-4, 1 - 1e-4)
    for t0, t1 in zip(steps[:-1], steps[1:]):
        dt = (t1 - t0).days
        qo0, qw0, qo4, qw4 = qo_hist[-1], qw_hist[-1], qo_hist[-5], qw_hist[-5]
        st = pd.DataFrame({"case_num": cases, "ln_qo": lg(qo0), "ln_qw": lg(qw0),
                           "logit_wc": logit(wcf(qo0, qw0)),
                           "D_recent": -(lg(qo0) - lg(qo4)),
                           "B_recent": logit(wcf(qo0, qw0)) - logit(wcf(qo4, qw4)),
                           "E_recent": lg(qw0) - lg(qw4), "rf": no / ooip, "wf": nw / pva,
                           "no_s": no / 1e7, "nw_s": nw / 1e7})
        X = design(st, geo, cols)
        E = np.clip(mE.predict(X), -1.0, 1.0)
        if oil == "roll":
            D = np.clip(mD.predict(X), 0.0, 2.0)
        else:
            tq = (t0 - cutoff).days / YEAR
            D = D0 / (1 + b * D0 * tq) if b > 1e-4 else D0
        qo1 = qo0 * np.exp(-D * dt / YEAR); qw1 = np.maximum(qw0, 1e-9) * np.exp(E * dt / YEAR)
        def integ(q0, q1):
            r = np.log(q1 / q0)
            return np.where(np.abs(r) > 1e-9, dt * (q1 - q0) / np.where(np.abs(r) > 1e-9, r, 1), dt * q0)
        dno = integ(qo0, qo1); dnw = integ(np.maximum(qw0, 1e-9), qw1)
        no, nw, ng = no + dno, nw + dnw, ng + gor * dno
        qo_hist = qo_hist[1:] + [qo1]; qw_hist = qw_hist[1:] + [qw1]
        res[t1] = np.column_stack([no, ng, nw])
    return np.stack([res[d] for d in dates], axis=1)


def run_roll(panel, geo, cols, kind, model_id, oil="roll", dev_cases=range(1, 86),
             train_cases=range(1, 71), seed=42):
    train_cases = list(train_cases)
    folds = {}
    for f, (_, hold) in enumerate(KFold(5, shuffle=True, random_state=seed).split(train_cases)):
        for i in hold:
            folds[train_cases[i]] = f
    out = []
    for origin, horizon in EXPERIMENTS:
        cutoff = pd.Timestamp(origin)
        dates = experiment_dates(origin, horizon)
        groups = {f: [c for c in train_cases if folds[c] == f] for f in range(5)}
        groups["val"] = [c for c in dev_cases if c > 70]
        for key, hold in groups.items():
            fit_cases = [c for c in train_cases if c not in hold] if key != "val" else train_cases
            tr = train_rows(panel, fit_cases, cutoff)
            tr = tr[tr.E_recent.notna() & tr.E_fwd.notna() & (tr.qw > 1)]
            X = design(tr, geo, cols)
            mD = make_model(kind).fit(X, tr.D_fwd.clip(lower=0))
            mE = make_model(kind).fit(X, tr.E_fwd)
            b = estimate_b(panel, fit_cases, cutoff)
            hists = [panel[(panel.case_num == c) & (panel.date <= cutoff)] for c in hold]
            D0 = None
            if oil == "hyperbolic":
                last = pd.concat([h.tail(1) for h in hists])
                D0 = np.clip(mD.predict(design(last, geo, cols)), 0, 2)
            cum = rollforward_batch(hists, geo, cols, mD, mE, dates, oil=oil, D0=D0, b=b)
            for i, c in enumerate(hold):
                for j, ph in enumerate(["oil_cum", "gas_cum", "water_cum"]):
                    out.append(pd.DataFrame({"model_id": model_id, "case_num": c,
                                             "origin": cutoff, "horizon_years": horizon,
                                             "date": dates, "phase": ph,
                                             "prediction": cum[i, :, j]}))
    return pd.concat(out, ignore_index=True)
