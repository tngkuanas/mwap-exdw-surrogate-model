"""hz_plots.py — figures for Hazeem's geology track (structure, backtest, 2028)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

BLUE_SEQ = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
GREY = LinearSegmentedColormap.from_list("g", ["#e8e6e1", "#e8e6e1"])


def fault_faces(grid_case):
    """Faces between neighbouring columns with a large depth jump (fault throw).
    Returns boolean arrays (I-faces, J-faces) shaped like the grid minus one."""
    Z = np.full((53, 55, 31), np.nan)
    Z[grid_case["I Index"] - 1, grid_case["J Index"] - 1, grid_case["K Index"] - 1] = grid_case["Z Coordinate"]
    th = np.nanmedian(np.abs(np.diff(Z, axis=2)))
    out = []
    for ax in (0, 1):
        d = np.abs(np.diff(Z, axis=ax))
        v = d[~np.isnan(d)]
        thr = max(5 * np.median(v), 0.5 * th)
        out.append(np.nan_to_num(d) > thr)
    return out


def plot_structure(grid_case, path=None):
    shape = (53, 55, 31)
    O = np.zeros(shape); A = np.zeros(shape, bool)
    ix = (grid_case["I Index"] - 1, grid_case["J Index"] - 1, grid_case["K Index"] - 1)
    O[ix] = grid_case["Pore Volume (PORV)"] * grid_case["Oil Saturation (SOIL)"]; A[ix] = True
    fI, fJ = fault_faces(grid_case)
    oil = O.sum(2) / 1e6; act = A.any(2)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axs[0]
    img = np.where(act, oil, np.nan); img[img == 0] = np.nan
    ext = [0.5, 53.5, 0.5, 55.5]
    ax.imshow(np.where(act, 0, np.nan).T, origin="lower", cmap=GREY, extent=ext)
    im = ax.imshow(img.T, origin="lower", cmap=BLUE_SEQ, extent=ext)
    ax.axvspan(50.5, 53.5, color="#9a9893", alpha=0.55, lw=0)
    ax.text(52, 28, "Aquifer cells (I = 51–53)", rotation=90, ha="center", va="center", fontsize=9)
    for i, j in np.argwhere(fI.sum(2) >= 5):
        ax.plot([i + 1.5, i + 1.5], [j + 0.5, j + 1.5], color="#c0392b", lw=2.2, solid_capstyle="butt")
    for i, j in np.argwhere(fJ.sum(2) >= 5):
        ax.plot([i + 0.5, i + 1.5], [j + 1.5, j + 1.5], color="#c0392b", lw=2.2, solid_capstyle="butt")
    ax.plot([], [], color="#c0392b", lw=2.2, label="Fault (depth jump in ≥5 layers)")
    fip = np.full(shape, np.nan); fip[ix] = grid_case["Fluid In Place Region (FIPNUM)"]
    ow, oe = O[fip == 1].sum() / 1e6, O[fip == 2].sum() / 1e6
    ax.text(2, 47, f"WEST block\nFIPNUM 1\n{ow:.0f} M rb oil ({ow/(ow+oe):.0%})", fontsize=10,
            bbox=dict(fc="white", ec="none", alpha=.85))
    ax.text(12, 2.2, f"EAST block · FIPNUM 2 · {oe:.0f} M rb oil ({oe/(ow+oe):.0%})", fontsize=10,
            bbox=dict(fc="white", ec="none", alpha=.85))
    ax.annotate("Main fault =\nFIPNUM boundary", xy=(18.5, 33), xytext=(3, 25), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=.8))
    ax.annotate("Secondary\nfault", xy=(37.5, 42), xytext=(41, 48), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=.8))
    ax.set_xlabel("I index (west → east)"); ax.set_ylabel("J index (south → north)")
    ax.set_title("Top view: oil in place per column", fontsize=12, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02); cb.set_label("Oil in place (million reservoir bbl)")
    ax.legend(loc="upper right", fontsize=9)
    ax = axs[1]
    k = np.arange(1, 32); oilk = O.sum((0, 1)) / 1e6
    col = ["#256abf" if kk <= 5 else ("#e8e6e1" if (6 <= kk <= 12 or kk == 20) else "#e08a1e") for kk in k]
    ax.barh(k, oilk, color=col, height=0.8)
    ax.invert_yaxis(); ax.set_xlabel("Oil in place (million reservoir bbl)"); ax.set_ylabel("Layer K (top → bottom)")
    ax.set_title("Oil by layer, and where the perm multiplier acts", fontsize=12, loc="left")
    z = lambda a, b: oilk[a - 1:b].sum()
    ax.text(15.5, 3, f"Zone 1 (K1–5): {z(1,5):.0f} M rb\nPerm multiplier does NOT apply", fontsize=9, va="center")
    ax.text(15.5, 9, "K6–12: barrier, no oil", fontsize=9, va="center", color="#6b6a66")
    ax.text(15.5, 16.5, f"Zone 2 (K13–19): {z(13,19):.0f} M rb", fontsize=9, va="center")
    ax.text(15.5, 20, "K20: barrier", fontsize=9, va="center", color="#6b6a66")
    ax.text(15.5, 26, f"Zone 3 (K21–31): {z(21,31):.0f} M rb\nPerm multiplier acts on the TIGHT\nfacies (~0.5 mD) in K13–31 only",
            fontsize=9, va="center")
    for a in axs:
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e8e6e1"); ax.set_axisbelow(True); ax.set_xlim(0, 30)
    fig.suptitle("Reservoir structure decoded from the grid files", fontsize=14, x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_backtest(pv, path=None):
    sel = [("Baseline_Anas_Separate_hist", "Baseline (Anas):\nhistory-fit exponential"),
           ("Ridge | production state (no grid)", "Ridge, production\nstate only"),
           ("Ridge | grid-normalised state", "Ridge, + grid\nnormalisation"),
           ("SVR | production state (no grid)", "SVR, production\nstate only"),
           ("SVR | grid-normalised state", "SVR, + grid\nnormalisation"),
           ("SVR | grid-normalised state + geology", "SVR, + grid norm.\n+ 9 geology features")]
    sel = [s for s in sel if s[0] in pv.index]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(sel)); w = 0.38
    m = [pv.loc[k, "mean"] for k, _ in sel]; wo = [pv.loc[k, "worst"] for k, _ in sel]
    ax.bar(x - w / 2 - 0.01, m, w, color=["#9a9893"] + ["#2a78d6"] * (len(sel) - 1))
    ax.bar(x + w / 2 + 0.01, wo, w, color=["#cfcdc8"] + ["#9ec5f4"] * (len(sel) - 1))
    for xi, v in zip(x - w / 2, m):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + w / 2, wo):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=9, color="#6b6a66")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in sel], fontsize=9)
    ax.set_ylabel("Increment NRMSE (lower is better)"); ax.set_ylim(0, max(wo) * 1.13)
    ax.axhline(m[0], color="#9a9893", lw=1, ls="--")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e8e6e1"); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color="#2a78d6", label="Mean of 4 experiments (darker bar)"),
                       Patch(color="#9ec5f4", label="Worst experiment (lighter bar)")],
              frameon=False, loc="upper right")
    ax.set_title("Backtest error: hyperbolic decline predicted from state + grid features", loc="left", fontsize=12)
    fig.text(0.01, -0.02, "Backtests: cutoffs 2003 (3y, 5y), 2004 (3y), 2005 (3y); cases 1-70 out-of-fold + "
             "71-85 validation; training uses only data before each cutoff. Scored with backtest_engine.py.",
             fontsize=8, color="#6b6a66")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def plot_forecast(F, panel, cases=(71, 78, 85), path=None):
    fig, axs = plt.subplots(1, len(cases), figsize=(14, 4.3), sharey=True)
    style = {"Baseline_Anas_Separate_hist": ("#9a9893", "Baseline (exponential, history fit)", "--"),
             "Hazeem_SVR_exponential": ("#2a78d6", "SVR decline, b = 0", "-"),
             "Hazeem_SVR_hyperbolic": ("#e08a1e", "SVR decline, b from data (hyperbolic)", "-")}
    for ax, c in zip(axs, cases):
        h = panel[panel.case_num == c]
        ax.plot(h.date, h["no"] / 1e6, color="#1f1f1d", lw=2, label="Simulated history")
        for mid, (cc, lab, ls) in style.items():
            f = F[(F.case_num == c) & (F.model_id == mid) & (F.phase == "oil_cum")]
            ax.plot(f.date, f.prediction / 1e6, color=cc, lw=2, ls=ls, label=lab)
        ax.axvline(pd.Timestamp("2008-01-01"), color="#cfcdc8", lw=1)
        ax.set_title(f"Case {c}", loc="left", fontsize=11)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.grid(color="#eeece8"); ax.set_axisbelow(True)
    axs[0].set_ylabel("Cumulative oil (million STB)")
    axs[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Oil forecast 2008 → 2028: the decline shape (b) is the biggest open choice",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig
