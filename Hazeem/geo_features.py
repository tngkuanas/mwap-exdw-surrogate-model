"""
geo_features.py — physics-based features engineered from the 3D grid (files 03-08).

What the grid decoding showed (see 01 notebook):
  * Porosity multiplier  -> scales PORO and PORV in every cell.
  * Permeability mult.   -> scales PERMX/TRANX/TRANY only in the TIGHT facies
                            (~0.5 mD cells, 11,575 cells in K13-31, ~19% of the oil).
                            Good rock (~100 mD), Zone 1 (K1-5) and PERMZ/TRANZ never change.
  * Aquifer pore volume  -> scales PORV of the 2,286 aquifer cells at I = 51-53.
  * Fault transmissibility -> NOT present in any exported grid column; the simulator
                            applies it on fault faces. We locate the main fault from
                            the FIPNUM 1/2 boundary (it coincides with the depth jumps)
                            and multiply the boundary transmissibility by the parameter.
  * SOIL / SWAT / PRESSURE -> identical in all 100 cases (initial state).

Every feature below is computed from the grid of each case, so it is valid for
validation and test cases too (no production labels are used).
"""
import numpy as np
import pandas as pd

PARAMS = ["Fault Transmissibility", "Porosity Multiplier",
          "Permeability Multiplier", "Aquifer Pore Volume"]

ZONES = {"Z1": (1, 5), "Z2": (13, 19), "Z3": (21, 31)}
AQUIFER_I = 51          # aquifer cells: I >= 51
LOWER_K = 13            # perm multiplier acts on K >= 13

COLS = {
    "PORV": "Pore Volume (PORV)", "PORO": "Porosity (PORO)",
    "PERMX": "Permeability I (PERMX)", "PERMZ": "Permeability K (PERMZ)",
    "TRANX": "Transmissibility I (TRANX)", "TRANY": "Transmissibility J (TRANY)",
    "TRANZ": "Transmissibility K (TRANZ)", "FIP": "Fluid In Place Region (FIPNUM)",
    "SOIL": "Oil Saturation (SOIL)", "SWAT": "Water Saturation (SWAT)",
}


def _cube(df, col, fill=np.nan):
    a = np.full((53, 55, 31), fill, dtype=float)
    a[df["I Index"].to_numpy() - 1, df["J Index"].to_numpy() - 1,
      df["K Index"].to_numpy() - 1] = df[col].to_numpy()
    return a


def tight_facies_mask(grid):
    """Cells whose PERMX differs between the supplied case grids = the tight facies
    that the permeability multiplier acts on. Uses grid inputs only (no labels).
    Returns a set of (I, J, K) tuples."""
    p = grid.pivot_table(index=["I Index", "J Index", "K Index"], columns="case_num",
                         values=COLS["PERMX"])
    varies = (p.max(axis=1) / p.min(axis=1)) > 1 + 1e-6
    return set(p.index[varies])


def case_features(df, tight=None):
    """Features for one case (df = that case's rows of grid_properties)."""
    k = df["K Index"].to_numpy()
    i = df["I Index"].to_numpy()
    porv = df[COLS["PORV"]].to_numpy()
    soil = df[COLS["SOIL"]].to_numpy()
    oil = porv * soil
    res = i < AQUIFER_I                       # reservoir (non-aquifer) cells
    f = {}

    # 1) Volumes: how much oil / how big the aquifer
    f["OOIP_total"] = oil.sum()
    for z, (k0, k1) in ZONES.items():
        f[f"OOIP_{z}"] = oil[(k >= k0) & (k <= k1)].sum()
    fip = df[COLS["FIP"]].to_numpy()
    f["OOIP_west"] = oil[fip == 1].sum()
    f["OOIP_east"] = oil[fip == 2].sum()
    f["PV_reservoir"] = porv[res].sum()
    f["PV_aquifer"] = porv[~res].sum()
    f["aquifer_to_oil_ratio"] = f["PV_aquifer"] / f["OOIP_total"]

    # 2) Flow capacity per zone (sum of horizontal transmissibility)
    tx = df[COLS["TRANX"]].to_numpy()
    ty = df[COLS["TRANY"]].to_numpy()
    th = tx + ty
    for z, (k0, k1) in ZONES.items():
        m = (k >= k0) & (k <= k1) & res
        f[f"T_{z}"] = th[m].sum()
        w = porv[m]
        f[f"PERMX_pvw_{z}"] = np.average(df[COLS["PERMX"]].to_numpy()[m], weights=w)
    f["T_total"] = f["T_Z1"] + f["T_Z2"] + f["T_Z3"]
    f["T_lower_fraction"] = (f["T_Z2"] + f["T_Z3"]) / f["T_total"]

    # 3) Aquifer connection: transmissibility of faces between I=50 and I=51
    T = _cube(df, COLS["TRANX"], 0.0)
    f["T_aquifer_connection"] = T[AQUIFER_I - 2, :, :].sum()   # face 50 -> 51

    # 4) Cross-fault transmissibility (main fault = FIPNUM 1|2 boundary)
    F = _cube(df, COLS["FIP"])
    TY = _cube(df, COLS["TRANY"], 0.0)
    bx = (F[:-1] != F[1:]) & ~np.isnan(F[:-1]) & ~np.isnan(F[1:])
    by = (F[:, :-1] != F[:, 1:]) & ~np.isnan(F[:, :-1]) & ~np.isnan(F[:, 1:])
    f["T_fault_faces_base"] = T[:-1][bx].sum() + TY[:, :-1][by].sum()

    # 5) Speed / diffusivity proxies (flow capacity per unit of stored oil)
    f["diffusivity_proxy"] = f["T_total"] / f["PV_reservoir"]
    f["drainage_time_proxy"] = f["OOIP_total"] / f["T_total"]

    # 6) Tight facies: the slow, late-time oil the perm multiplier controls
    if tight is not None:
        key = list(zip(df["I Index"], df["J Index"], df["K Index"]))
        m = np.fromiter((x in tight for x in key), bool, len(key))
        f["OOIP_tight"] = oil[m].sum()
        f["OOIP_tight_fraction"] = f["OOIP_tight"] / f["OOIP_total"]
        f["T_tight"] = th[m].sum()
        f["PERMX_pvw_tight"] = np.average(df[COLS["PERMX"]].to_numpy()[m], weights=porv[m])
        f["tight_drainage_time"] = f["OOIP_tight"] / f["T_tight"]
        f["OOIP_good"] = f["OOIP_total"] - f["OOIP_tight"]
    return f


def build_features(grid, unc):
    """grid: full grid_properties frame; unc: uncertainty_params frame."""
    tight = tight_facies_mask(grid)
    rows = []
    for case, df in grid.groupby("case_num", sort=True):
        r = case_features(df, tight)
        r["case_num"] = int(case)
        rows.append(r)
    feats = pd.DataFrame(rows).set_index("case_num")
    u = unc.set_index("case_num")[PARAMS]
    # the fault multiplier lives outside the grid -> apply it to the fault faces
    feats["T_cross_fault"] = feats["T_fault_faces_base"] * u["Fault Transmissibility"]
    feats["west_connectivity"] = feats["T_cross_fault"] / feats["OOIP_west"]
    feats["aquifer_drive_index"] = feats["PV_aquifer"] * feats["T_aquifer_connection"] / feats["OOIP_total"]
    return feats.join(u)
