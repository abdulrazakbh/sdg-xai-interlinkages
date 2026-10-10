"""
SDG infrastructure-interlinkage paper - Step 2 (version 2): explainable ML models
================================================================================

Changes from version 1
  * World-region dummies added as controls (absorbs geography such as desert dust).
  * Internet users (17.8.1) excluded from the 3G-coverage model (near-mechanical link);
    energy intensity (7.3.1) still excluded from the CO2-per-GDP model.
  * Manufacturing share (9.2.1) moved to a supplementary model (no explanatory power in v1).
  * New CHANGE models: each country's 5-year change in an outcome, explained by baseline
    levels, 5-year driver changes, the baseline outcome level (convergence) and GDP growth.
  * Country-cluster bootstrap (B resamples of countries) gives 95% intervals on every
    driver's importance share and the stability of its direction.
  * Ablation: leave-country-out R2 with and without the region controls.

Input : panel_wide.csv from sdg_coverage_check.py (2010-2023, corrected series)
Output: results_v2/  (tables, 300-dpi figures)

Run:  pip install pandas numpy scipy scikit-learn xgboost shap matplotlib statsmodels
      python sdg_xai_model_v2.py path/to/panel_wide.csv [n_bootstrap]
"""

from __future__ import annotations

import sys
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 42
RNG = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------

OUTCOMES = {                       # code: (label, direction; +1 = higher is better)
    "11.1.1": ("Urban slum population (%)", -1),
    "11.6.2": ("Urban PM2.5 (µg/m³)", -1),
    "9.4.1":  ("CO₂ per unit GDP (log kg/$)", -1),
    "9.c.1":  ("3G network coverage (%)", +1),
}
SUPPLEMENTARY = {"9.2.1": ("Manufacturing VA (% GDP)", +1)}
ALL_OUT = {**OUTCOMES, **SUPPLEMENTARY}

DRIVERS = {
    "6.1.1":  ("Safe drinking water", +1),
    "6.2.1":  ("Safe sanitation", +1),
    "6.4.1":  ("Water-use efficiency", +1),
    "6.4.2":  ("Water stress", -1),
    "7.1.1":  ("Electricity access", +1),
    "7.1.2":  ("Clean cooking fuels", +1),
    "7.2.1":  ("Renewable energy share", +1),
    "7.3.1":  ("Energy intensity", -1),
    "17.6.1": ("Fixed broadband", +1),
    "17.8.1": ("Internet users", +1),
}
EXCLUDE = {"9.4.1": ["7.3.1"], "9.c.1": ["17.8.1"]}
# Change models with no out-of-country skill (LOCO R2 < 0.1 in testing): reported, not interpreted
CHANGE_UNRELIABLE = {"11.1.1"}
LOG_VARS = ["6.4.1", "6.4.2", "9.4.1"]

REGION_SHORT = {
    "Europe & Central Asia": "ECA", "Sub-Saharan Africa": "SSA",
    "Latin America & Caribbean": "LAC", "East Asia & Pacific": "EAP",
    "Middle East, North Africa, Afghanistan & Pakistan": "MENAAP",
    "South Asia": "SAS", "North America": "NAM",
}
REGION_COLS = [f"reg_{v}" for v in REGION_SHORT.values()]

LABEL = {k: v[0] for k, v in {**ALL_OUT, **DRIVERS}.items()}
LABEL.update({"log_gdp_pc": "GDP per capita (log)", "urban_pct": "Urbanisation (%)",
              "log_pop": "Population (log)", "region": "World region",
              "gdp_growth": "GDP per capita growth (Δlog)", "base_outcome": "Baseline outcome level"})
for c in DRIVERS:
    LABEL[f"{c}_base"] = f"{LABEL[c]} (baseline)"
    LABEL[f"{c}_chg"] = f"Δ {LABEL[c]}"
DIRECTION = {k: v[1] for k, v in {**ALL_OUT, **DRIVERS}.items()}

INCOME_ORDER = ["Low income", "Lower middle income", "Upper middle income", "High income"]
# Regularised after a leave-country-out comparison of depth 2-4 settings
XGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, max_depth=3, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                  random_state=SEED, n_jobs=4)

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_levels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["year"].between(2010, 2023)].copy()
    df.loc[df["7.3.1"] <= 0, "7.3.1"] = np.nan
    df["log_gdp_pc"] = np.log(df["gdp_pc_const2015usd"])
    df["log_pop"] = np.log(df["population"])
    cols = list(ALL_OUT) + list(DRIVERS)
    df = df.sort_values(["iso3", "year"])
    df[cols] = df.groupby("iso3")[cols].transform(
        lambda s: s.interpolate(limit=3, limit_area="inside"))
    for c in LOG_VARS:
        df[c] = np.log(df[c].where(df[c] > 0))
    for name, short in REGION_SHORT.items():
        df[f"reg_{short}"] = (df["region"] == name).astype(float)
    return df


def build_changes(lev: pd.DataFrame, outcome: str, horizon: int = 5) -> pd.DataFrame:
    """One row per country and end-year t (2015-2023): change from t-5 to t."""
    a = lev.set_index(["iso3", "year"])
    rows = []
    for (iso, t), r in a.iterrows():
        if t - horizon < 2010 or (iso, t - horizon) not in a.index:
            continue
        b = a.loc[(iso, t - horizon)]
        if pd.isna(r[outcome]) or pd.isna(b[outcome]) or pd.isna(b["log_gdp_pc"]):
            continue
        row = {"iso3": iso, "year": t, "income_group": r["income_group"],
               "y": r[outcome] - b[outcome], "base_outcome": b[outcome],
               "log_gdp_pc": b["log_gdp_pc"], "urban_pct": b["urban_pct"],
               "log_pop": b["log_pop"], "gdp_growth": r["log_gdp_pc"] - b["log_gdp_pc"]}
        for c in DRIVERS:
            row[f"{c}_base"] = b[c]
            row[f"{c}_chg"] = r[c] - b[c]
        for rc in REGION_COLS:
            row[rc] = b[rc]
        rows.append(row)
    return pd.DataFrame(rows)


def level_features(outcome: str, region: bool = True) -> list[str]:
    drv = [d for d in DRIVERS if d not in EXCLUDE.get(outcome, [])]
    return drv + ["log_gdp_pc", "urban_pct", "log_pop"] + (REGION_COLS if region else [])


def change_features(outcome: str) -> list[str]:
    drv = [d for d in DRIVERS if d not in EXCLUDE.get(outcome, [])]
    return ([f"{d}_base" for d in drv] + [f"{d}_chg" for d in drv]
            + ["base_outcome", "log_gdp_pc", "gdp_growth", "urban_pct", "log_pop"] + REGION_COLS)

# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def metrics(y, p):
    return {"R2": r2_score(y, p), "RMSE": mean_squared_error(y, p) ** 0.5,
            "MAE": mean_absolute_error(y, p)}


MODELS = {
    "XGBoost": lambda: xgb.XGBRegressor(**XGB_PARAMS),
    "RandomForest": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                          RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                                random_state=SEED, n_jobs=4)),
    "Ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   RidgeCV(alphas=np.logspace(-3, 3, 13))),
}


def loco(X, y, groups, build) -> dict:
    pred = np.zeros_like(y, dtype=float)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = build(); m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return metrics(y, pred)

# ----------------------------------------------------------------------------
# SHAP helpers
# ----------------------------------------------------------------------------

def collapse_region(sv: np.ndarray, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Sum region-dummy SHAP columns into one 'region' feature."""
    keep = [i for i, c in enumerate(cols) if c not in REGION_COLS]
    reg = [i for i, c in enumerate(cols) if c in REGION_COLS]
    out = sv[:, keep]
    names = [cols[i] for i in keep]
    if reg:
        out = np.column_stack([out, sv[:, reg].sum(axis=1)])
        names.append("region")
    return out, names


def driver_of(col: str) -> str | None:
    base = col.replace("_base", "").replace("_chg", "")
    return base if base in DRIVERS else None


def shares_and_signs(sv, X: pd.DataFrame, cols, outcome, focus: list[str]):
    """Importance share among `focus` columns and aligned direction (+1 synergy, -1 trade-off)."""
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=cols)
    share = mean_abs[focus] / mean_abs[focus].sum()
    sign = {}
    for c in focus:
        j = cols.index(c)
        x = X[c].values.astype(float)
        ok = ~np.isnan(x)
        rho = spearmanr(x[ok], sv[ok, j]).correlation if ok.sum() > 10 else np.nan
        raw = np.sign(rho) if (rho == rho and abs(rho) > 0.1) else 0.0
        d = driver_of(c)
        sign[c] = raw * DIRECTION[d] * DIRECTION[outcome] if d else raw
    return share, pd.Series(sign), mean_abs


def bootstrap(d: pd.DataFrame, cols, ycol, outcome, focus, B):
    countries = d["iso3"].unique()
    shares, signs = [], []
    for b in range(B):
        pick = RNG.choice(countries, size=len(countries), replace=True)
        db = pd.concat([d[d["iso3"] == c] for c in pick], ignore_index=True)
        m = xgb.XGBRegressor(**{**XGB_PARAMS, "random_state": SEED + b}).fit(db[cols].values, db[ycol].values)
        sv = shap.TreeExplainer(m).shap_values(db[cols].values)
        s, g, _ = shares_and_signs(sv, db[cols], cols, outcome, focus)
        shares.append(s); signs.append(g)
    S, G = pd.DataFrame(shares), pd.DataFrame(signs)
    return S, G

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

INK, MUTED, GRID = "#1f2328", "#5b6470", "#d0d7de"
SYN, TRADE, NEU = "#2a6fb0", "#c0392b", "#8c959f"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def wrap(s, n=16):
    return "\n".join(textwrap.wrap(s, n))


def fig_beeswarm(sv, X, cols, title, out):
    svc, names = collapse_region(sv, cols)
    Xc = X[[c for c in cols if c not in REGION_COLS]].copy()
    if "region" in names:
        Xc["region"] = np.nan
    plt.figure()
    shap.summary_plot(svc, Xc.values, feature_names=[LABEL[c] for c in names], show=False,
                      max_display=14, plot_size=(7, 5.2))
    plt.title(title, fontsize=10, loc="left")
    plt.savefig(out); plt.close("all")


def fig_matrix(tab: pd.DataFrame, value_col: str, title: str, out: Path, outs: list[str]):
    """tab: rows = driver codes x outcome, columns share, sign, stable."""
    drv = list(DRIVERS)
    share = tab.pivot(index="driver", columns="outcome", values=value_col).reindex(index=drv, columns=outs)
    sign = tab.pivot(index="driver", columns="outcome", values="sign").reindex(index=drv, columns=outs)
    stable = tab.pivot(index="driver", columns="outcome", values="sign_stability").reindex(index=drv, columns=outs)
    val = share * sign
    lim = 0.35
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(val.values.astype(float), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(outs))); ax.set_xticklabels([f"{c}\n{wrap(LABEL[c], 14)}" for c in outs], fontsize=7.5)
    ax.set_yticks(range(len(drv))); ax.set_yticklabels([f"{c}  {LABEL[c]}" for c in drv])
    for i in range(len(drv)):
        for j in range(len(outs)):
            v = val.values[i, j]
            if pd.isna(v):
                ax.text(j, i, "excl.", ha="center", va="center", fontsize=7, color=MUTED)
                continue
            mark = "" if (stable.values[i, j] >= 0.8) else "†"
            ax.text(j, i, f"{share.values[i, j]*100:.0f}%{mark}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > lim * 0.6 else INK)
    ax.set_xticks(np.arange(-.5, len(outs)), minor=True); ax.set_yticks(np.arange(-.5, len(drv)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("← trade-off    importance share × direction    synergy →")
    ax.set_title(title + "\nCell = share of driver importance; † = direction stable in <80% of bootstrap draws",
                 fontsize=9, loc="left")
    fig.savefig(out); plt.close(fig)


def fig_forest(tab: pd.DataFrame, out: Path, title: str, outs: list[str]):
    fig, axes = plt.subplots(1, len(outs), figsize=(13, 4.2), sharey=True)
    drv = list(DRIVERS)
    for ax, oc in zip(axes, outs):
        t = tab[tab["outcome"] == oc].set_index("driver").reindex(drv)
        y = np.arange(len(drv))
        for k, (dname, r) in enumerate(t.iterrows()):
            if pd.isna(r["share"]):
                ax.text(0.005, k, "excluded", va="center", fontsize=7, color=MUTED); continue
            c = SYN if r["sign"] > 0 else TRADE if r["sign"] < 0 else NEU
            ax.plot([r["ci_low"] * 100, r["ci_high"] * 100], [k, k], color=c, linewidth=2, solid_capstyle="round")
            ax.plot(r["share"] * 100, k, "o", color=c, markersize=5, markeredgecolor="white", markeredgewidth=1)
        ax.set_yticks(y); ax.set_yticklabels([LABEL[c] for c in drv])
        ax.set_title(f"{oc}\n{LABEL[oc]}", fontsize=8.5, loc="left")
        ax.set_xlabel("% of driver importance (95% CI)")
        ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        ax.set_xlim(left=0)
    axes[0].invert_yaxis()   # shared y: invert once so drivers read top-down in the same order
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=SYN, marker="o", label="Synergy"),
               Line2D([0], [0], color=TRADE, marker="o", label="Trade-off"),
               Line2D([0], [0], color=NEU, marker="o", label="No clear direction")]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(title, x=0.06, ha="left", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out); plt.close(fig)


def fig_dependence(x, s, feat_label, out_label, out, log=False):
    ok = ~np.isnan(x)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.scatter(x[ok], s[ok], s=6, alpha=0.35, color=SYN, linewidths=0)
    thr = None
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        lw = lowess(s[ok], x[ok], frac=0.3)
        ax.plot(lw[:, 0], lw[:, 1], color=INK, linewidth=2)
        thr = plateau(lw)
        if thr is not None:
            ax.axvline(thr, color=TRADE, linestyle="--", linewidth=1)
            ax.text(thr, ax.get_ylim()[1], f" effect flattens ≈ {thr:.3g}", color=TRADE, fontsize=7.5, va="top")
    except Exception:
        pass
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_xlabel(feat_label + (" (log)" if log else "")); ax.set_ylabel("SHAP value")
    ax.set_title(f"{feat_label} → {out_label}", fontsize=9, loc="left")
    fig.savefig(out); plt.close(fig)
    return thr


def plateau(lw):
    x, y = lw[:, 0], lw[:, 1]
    _, idx = np.unique(x, return_index=True); x, y = x[idx], y[idx]
    if len(x) < 20: return None
    grid = np.linspace(x.min(), x.max(), 200); yy = np.interp(grid, x, y)
    slope = np.gradient(yy, grid); peak = np.nanmax(np.abs(slope))
    if peak == 0: return None
    i = int(np.nanargmax(np.abs(slope)))
    flat = np.where(np.abs(slope[i:]) < 0.15 * peak)[0]
    if len(flat) == 0: return None
    t = grid[i + flat[0]]
    return float(t) if t < np.quantile(grid, 0.95) else None


def fig_india(sv_row, data_row, cols, base, title, out, direction):
    svc, names = collapse_region(sv_row[None, :], cols)
    vals = pd.Series(svc[0], index=names)
    top = vals.reindex(vals.abs().sort_values(ascending=False).index)[:10][::-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    colors = [SYN if v * direction > 0 else TRADE for v in top.values]  # blue = pushes outcome in the better direction
    ax.barh(range(len(top)), top.values, color=colors, height=0.6)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([LABEL[c] for c in top.index])
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel(f"Contribution to prediction (base value {base:.2f})")
    ax.set_title(title, fontsize=9.5, loc="left")
    fig.savefig(out); plt.close(fig)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run(panel_path: str, B: int):
    t0 = time.time()
    out = Path(__file__).resolve().parent / "results_v2"
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    lev = load_levels(Path(panel_path))

    perf, lv_rows, ch_rows, samples, thr_rows, inc_rows, india_rows, corr_rows = [], [], [], [], [], [], [], []

    for oc in ALL_OUT:
        supp = oc in SUPPLEMENTARY
        print(f"\n=== {oc} {LABEL[oc]}{' (supplementary)' if supp else ''}")

        # ---------------- LEVEL MODEL ----------------
        cols = level_features(oc)
        d = lev.dropna(subset=[oc, "log_gdp_pc"]).reset_index(drop=True)
        X, y, g = d[cols].values, d[oc].values, d["iso3"].values
        samples.append({"outcome": oc, "model": "level", "observations": len(d), "countries": d["iso3"].nunique()})
        for name, build in MODELS.items():
            perf.append({"outcome": oc, "model_type": "level", "algorithm": name,
                         "controls": "with region", **loco(X, y, g, build)})
        perf.append({"outcome": oc, "model_type": "level", "algorithm": "XGBoost", "controls": "without region",
                     **loco(d[level_features(oc, region=False)].values, y, g, MODELS["XGBoost"])})

        m = xgb.XGBRegressor(**XGB_PARAMS).fit(X, y)
        ex = shap.TreeExplainer(m); sv = ex.shap_values(X)
        focus = [c for c in cols if c in DRIVERS]
        share, sign, mean_abs = shares_and_signs(sv, d[cols], cols, oc, focus)
        print(f"  level: n={len(d)}  LOCO R2={perf[-4]['R2']:.3f}  bootstrap B={B} ...", flush=True)
        S, G = bootstrap(d, cols, oc, oc, focus, B) if (B and not supp) else (None, None)
        for c in focus:
            r = {"outcome": oc, "driver": c, "share": share[c], "sign": sign[c]}
            if S is not None:
                r.update(ci_low=S[c].quantile(0.025), ci_high=S[c].quantile(0.975),
                         sign_stability=(G[c] == sign[c]).mean())
            lv_rows.append(r)
            ok = d[[c, oc]].dropna()
            rho = spearmanr(ok[c], ok[oc]).correlation
            corr_rows.append({"outcome": oc, "driver": c, "spearman_r": rho,
                              "corr_direction": np.sign(rho) * DIRECTION[c] * DIRECTION[oc] if abs(rho) > 0.1 else 0.0,
                              "shap_direction": sign[c], "shap_share": share[c]})
        ctrl = mean_abs.drop(focus)
        reg_share = mean_abs[REGION_COLS].sum() / mean_abs.sum()
        lv_rows.append({"outcome": oc, "driver": "_controls_share", "share": ctrl.sum() / mean_abs.sum(), "sign": np.nan})
        lv_rows.append({"outcome": oc, "driver": "_region_share", "share": reg_share, "sign": np.nan})

        fig_beeswarm(sv, d[cols], cols, f"{oc}  {LABEL[oc]}: level model", fig_dir / f"level_beeswarm_{oc}.png")

        for feat in share.sort_values(ascending=False).index[:2]:
            j = cols.index(feat)
            t = fig_dependence(d[feat].values.astype(float), sv[:, j], LABEL[feat], LABEL[oc],
                               fig_dir / f"level_dependence_{oc}_{feat}.png", log=feat in LOG_VARS)
            thr_rows.append({"outcome": oc, "model": "level", "driver": feat, "label": LABEL[feat],
                             "effect_flattens_at": t, "scale": "log" if feat in LOG_VARS else "raw"})

        for grp in INCOME_ORDER:
            mk = (d["income_group"] == grp).values
            if mk.sum() < 30: continue
            ga = pd.Series(np.abs(sv[mk]).mean(axis=0), index=cols)[focus]
            for c, v in (ga / ga.sum()).items():
                inc_rows.append({"outcome": oc, "income_group": grp, "driver": c, "share": v, "n": int(mk.sum())})

        ind = d.index[d["iso3"] == "IND"]
        if len(ind):
            r = ind[d.loc[ind, "year"].values.argmax()]
            yr = int(d.loc[r, "year"])
            fig_india(sv[r], d.loc[r, cols], cols, float(ex.expected_value),
                      f"India {yr}: {LABEL[oc]} (level model)", fig_dir / f"india_level_{oc}.png", DIRECTION[oc])
            svc, names = collapse_region(sv[r][None, :], cols)
            for j, c in enumerate(names):
                india_rows.append({"outcome": oc, "model": "level", "year": yr, "feature": c, "label": LABEL[c],
                                   "value": (np.nan if c == "region" else d.loc[r, c]), "shap": svc[0, j],
                                   "observed": d.loc[r, oc], "base_value": float(ex.expected_value)})

        # ---------------- CHANGE MODEL ----------------
        dc = build_changes(lev, oc).reset_index(drop=True)
        ccols = change_features(oc)
        Xc, yc, gc = dc[ccols].values, dc["y"].values, dc["iso3"].values
        samples.append({"outcome": oc, "model": "change (5-yr)", "observations": len(dc), "countries": dc["iso3"].nunique()})
        for name, build in MODELS.items():
            perf.append({"outcome": oc, "model_type": "change (5-yr)", "algorithm": name,
                         "controls": "with region", **loco(Xc, yc, gc, build)})
        mc = xgb.XGBRegressor(**XGB_PARAMS).fit(Xc, yc)
        exc = shap.TreeExplainer(mc); svc_ = exc.shap_values(Xc)
        cfocus = [c for c in ccols if c.endswith("_chg")]
        cshare, csign, cmean = shares_and_signs(svc_, dc[ccols], ccols, oc, cfocus)
        print(f"  change: n={len(dc)}  LOCO R2={perf[-1 - 2]['R2']:.3f}  bootstrap ...", flush=True)
        S2, G2 = (bootstrap(dc, ccols, "y", oc, cfocus, B)
                  if (B and not supp and oc not in CHANGE_UNRELIABLE) else (None, None))
        for c in cfocus:
            r = {"outcome": oc, "driver": driver_of(c), "share": cshare[c], "sign": csign[c]}
            if S2 is not None:
                r.update(ci_low=S2[c].quantile(0.025), ci_high=S2[c].quantile(0.975),
                         sign_stability=(G2[c] == csign[c]).mean())
            ch_rows.append(r)
        total = cmean.sum()
        groups = {"driver changes": cmean[cfocus].sum(),
                  "driver baselines": cmean[[c for c in ccols if c.endswith("_base")]].sum(),
                  "baseline outcome": cmean["base_outcome"],
                  "GDP level and growth": cmean[["log_gdp_pc", "gdp_growth"]].sum(),
                  "other controls": cmean[["urban_pct", "log_pop"] + REGION_COLS].sum()}
        for k, v in groups.items():
            ch_rows.append({"outcome": oc, "driver": f"_group:{k}", "share": v / total, "sign": np.nan})

        fig_beeswarm(svc_, dc[ccols], ccols, f"{oc}  {LABEL[oc]}: 5-year change model",
                     fig_dir / f"change_beeswarm_{oc}.png")

        ind = dc.index[(dc["iso3"] == "IND")]
        if len(ind):
            r = ind[dc.loc[ind, "year"].values.argmax()]
            yr = int(dc.loc[r, "year"])
            fig_india(svc_[r], dc.loc[r, ccols], ccols, float(exc.expected_value),
                      f"India {yr-5}–{yr}: change in {LABEL[oc]}", fig_dir / f"india_change_{oc}.png", DIRECTION[oc])
            s2, names = collapse_region(svc_[r][None, :], ccols)
            for j, c in enumerate(names):
                india_rows.append({"outcome": oc, "model": "change (5-yr)", "year": yr, "feature": c,
                                   "label": LABEL[c], "value": (np.nan if c == "region" else dc.loc[r, c]),
                                   "shap": s2[0, j], "observed": dc.loc[r, "y"],
                                   "base_value": float(exc.expected_value)})
        print(f"  done ({time.time() - t0:.0f}s elapsed)", flush=True)

    # ---------------- Save ----------------
    perf = pd.DataFrame(perf); perf.to_csv(out / "model_performance.csv", index=False)
    lv = pd.DataFrame(lv_rows); lv.to_csv(out / "level_driver_importance.csv", index=False)
    ch = pd.DataFrame(ch_rows); ch.to_csv(out / "change_driver_importance.csv", index=False)
    pd.DataFrame(samples).to_csv(out / "sample_sizes.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(out / "thresholds.csv", index=False)
    pd.DataFrame(inc_rows).to_csv(out / "income_group_importance.csv", index=False)
    pd.DataFrame(india_rows).to_csv(out / "india_local_shap.csv", index=False)
    corr = pd.DataFrame(corr_rows); corr["agree"] = corr["corr_direction"] == corr["shap_direction"]
    corr.to_csv(out / "correlation_vs_shap.csv", index=False)

    main_lv = lv[lv["outcome"].isin(OUTCOMES) & ~lv["driver"].str.startswith("_")]
    ch_outs = [o for o in OUTCOMES if o not in CHANGE_UNRELIABLE]
    main_ch = ch[ch["outcome"].isin(ch_outs) & ~ch["driver"].str.startswith("_")]
    if B:
        fig_matrix(main_lv, "share", "Level model: synergies (blue) and trade-offs (red)", fig_dir / "matrix_level.png", list(OUTCOMES))
        fig_matrix(main_ch, "share", "5-year change model: which driver improvements accompany progress",
                   fig_dir / "matrix_change.png", ch_outs)
        fig_forest(main_lv, fig_dir / "forest_level.png", "Driver importance in the level models, with 95% country-bootstrap intervals", list(OUTCOMES))
        fig_forest(main_ch, fig_dir / "forest_change.png", "Driver-change importance in the 5-year change models, with 95% country-bootstrap intervals", ch_outs)
    print(f"\nFinished in {time.time() - t0:.0f}s. Results in {out}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "panel_wide.csv",
        int(sys.argv[2]) if len(sys.argv) > 2 else 100)
