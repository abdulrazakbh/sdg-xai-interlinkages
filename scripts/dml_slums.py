"""
SDG infrastructure-interlinkage paper - Step 3: double machine learning check
=============================================================================

Question: holding income, urbanisation, population, region, year and the other SDG
drivers fixed, how much does urban slum share (11.1.1) change per unit of
  * electricity access (7.1.1, % of population)
  * clean cooking fuels (7.1.2, % of population)
  * fixed broadband (17.6.1, subscriptions per 100 people)?

Estimator: partially linear model  Y = theta*D + g(X) + e,  D = m(X) + v
(Chernozhukov et al., 2018). Nuisance functions g and m are learned with
cross-fitting over country-grouped folds, so a country's own data never
predicts its residuals. Standard errors are clustered by country. Each estimate
is the median over R repeated fold splits, with the DML median-aggregated SE.

Robustness
  1. Nuisance learner: XGBoost (main), random forest, lasso (linear benchmark).
  2. Within-country variation only: all variables demeaned by country.
  3. Broadband as log(1 + subscriptions), since SHAP showed the effect flattening.
  4. Placebo: see dml_block_placebo.py (block permutation of whole country series).
  5. Identification strength: share of treatment variance explained by controls.

Run:  python dml_slums.py path/to/panel_wide.csv [repetitions]
"""

from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v2", HERE / "sdg_xai_model_v2.py")
v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)

OUTCOME = "11.1.1"
TREATMENTS = {
    "7.1.1":  ("Electricity access", "percentage points", 10),
    "7.1.2":  ("Clean cooking fuels", "percentage points", 10),
    "17.6.1": ("Fixed broadband", "subscriptions per 100 people", 10),
}
SEED = 7

LEARNERS = {
    "XGBoost": lambda s: xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                                          subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                                          reg_lambda=5.0, random_state=s, n_jobs=4),
    "RandomForest": lambda s: make_pipeline(SimpleImputer(strategy="median"),
                                            RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                                  random_state=s, n_jobs=4)),
    "Lasso": lambda s: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                     LassoCV(cv=5, random_state=s, alphas=40)),
}


def controls_for(treat: str) -> list[str]:
    drv = [d for d in v2.DRIVERS if d != treat]
    return drv + ["log_gdp_pc", "urban_pct", "log_pop", "year"] + v2.REGION_COLS


def shuffled_groups(groups: np.ndarray, rng) -> np.ndarray:
    """Relabel countries randomly so GroupKFold gives a different split each repetition."""
    u = np.unique(groups)
    perm = dict(zip(u, rng.permutation(len(u))))
    return np.array([perm[g] for g in groups])


def dml_plr(y, d, X, groups, learner, R=10, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    thetas, ses, r2_m, r2_l = [], [], [], []
    for r in range(R):
        g_r = shuffled_groups(groups, rng)
        yhat, dhat = np.zeros_like(y), np.zeros_like(d)
        for tr, te in GroupKFold(n_splits=k).split(X, y, g_r):
            ml = LEARNERS[learner](seed + r); ml.fit(X[tr], y[tr]); yhat[te] = ml.predict(X[te])
            mm = LEARNERS[learner](seed + 100 + r); mm.fit(X[tr], d[tr]); dhat[te] = mm.predict(X[te])
        yr, dr = y - yhat, d - dhat
        theta = (dr @ yr) / (dr @ dr)
        psi = (yr - theta * dr) * dr
        J = np.mean(dr ** 2)
        n = len(y)
        s = pd.Series(psi).groupby(groups).sum().values          # cluster sums by country
        var = (s ** 2).sum() / (n ** 2 * J ** 2)
        thetas.append(theta); ses.append(np.sqrt(var))
        r2_m.append(1 - np.var(dr) / np.var(d)); r2_l.append(1 - np.var(yr) / np.var(y))
    thetas, ses = np.array(thetas), np.array(ses)
    t_med = np.median(thetas)
    se_med = np.sqrt(np.median(ses ** 2 + (thetas - t_med) ** 2))
    return {"theta": t_med, "se": se_med, "ci_low": t_med - 1.96 * se_med, "ci_high": t_med + 1.96 * se_med,
            "p_value": 2 * (1 - norm.cdf(abs(t_med / se_med))), "theta_min": thetas.min(), "theta_max": thetas.max(),
            "r2_treatment_on_controls": np.median(r2_m), "r2_outcome_on_controls": np.median(r2_l)}


def demean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = df[cols] - df.groupby("iso3")[cols].transform("mean")
    return out


def main(panel: str, R: int):
    t0 = time.time()
    out = HERE / "results_dml"; out.mkdir(exist_ok=True)
    lev = v2.load_levels(Path(panel))
    rows = []

    def record(spec_name, treat, res, n, n_c, scale):
        lab, unit, per = TREATMENTS[treat]
        k = per if scale == 1 else 1
        raw = {f"raw_{key}": val for key, val in res.items()}
        rows.append({"specification": spec_name, "treatment": treat, "label": lab, "n": n, "countries": n_c,
                     "effect_per": f"{per} {unit}" if scale == 1 else "1 log point",
                     "effect": res["theta"] * k, "ci_low": res["ci_low"] * k, "ci_high": res["ci_high"] * k,
                     "p_value": res["p_value"], "r2_treatment_on_controls": res["r2_treatment_on_controls"],
                     "r2_outcome_on_controls": res["r2_outcome_on_controls"], **raw})
        print(f"  {spec_name:<34} {treat:<7} effect={rows[-1]['effect']:+.2f} "
              f"[{rows[-1]['ci_low']:+.2f}, {rows[-1]['ci_high']:+.2f}]  p={res['p_value']:.3f}  "
              f"R2(D|X)={res['r2_treatment_on_controls']:.2f}  ({time.time() - t0:.0f}s)", flush=True)

    for treat in TREATMENTS:
        print(f"\n=== Treatment {treat} {TREATMENTS[treat][0]} -> {OUTCOME} slum share")
        X_cols = controls_for(treat)
        d = lev.dropna(subset=[OUTCOME, treat, "log_gdp_pc"]).reset_index(drop=True)
        y, D, g = d[OUTCOME].values, d[treat].values.astype(float), d["iso3"].values
        n, nc = len(d), d["iso3"].nunique()

        # Main + learner robustness
        for lrn in LEARNERS:
            record(f"Cross-country, {lrn} nuisance", treat, dml_plr(y, D, d[X_cols].values, g, lrn, R), n, nc, 1)

        # Within-country (country-demeaned) variation
        dm = demean(d, [OUTCOME, treat] + [c for c in X_cols if c not in v2.REGION_COLS])
        Xw = [c for c in X_cols if c not in v2.REGION_COLS]    # region is constant within country
        record("Within-country, XGBoost nuisance", treat,
               dml_plr(dm[OUTCOME].values, dm[treat].values, dm[Xw].values, g, "XGBoost", R), n, nc, 1)


    # Broadband in logs (non-linear effect seen in SHAP)
    treat = "17.6.1"
    print(f"\n=== Treatment log(1 + broadband) -> slum share")
    d = lev.dropna(subset=[OUTCOME, treat, "log_gdp_pc"]).reset_index(drop=True)
    record("Cross-country, log(1+broadband)", treat,
           dml_plr(d[OUTCOME].values, np.log1p(d[treat].values), d[controls_for(treat)].values,
                   d["iso3"].values, "XGBoost", R), len(d), d["iso3"].nunique(), scale=0)

    res = pd.DataFrame(rows)
    res.to_csv(out / "dml_results.csv", index=False)
    # Figure 6 is drawn by dml_block_placebo.py, which adds the placebo band.
    print(f"\nFinished in {time.time() - t0:.0f}s. Results in {out}")


def plot(res: pd.DataFrame, perm: pd.DataFrame, path: Path):
    INK, MUTED, GRID, MAIN, ALT = "#1f2328", "#5b6470", "#d0d7de", "#2a6fb0", "#8c959f"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "savefig.dpi": 300, "savefig.bbox": "tight"})
    specs = ["Cross-country, XGBoost nuisance", "Cross-country, RandomForest nuisance",
             "Cross-country, Lasso nuisance", "Within-country, XGBoost nuisance"]
    short = ["XGBoost (main)", "Random forest", "Lasso (linear)", "Within-country", "Placebo null (bias check)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, treat in zip(axes, TREATMENTS):
        t = res[(res["treatment"] == treat) & (res["specification"].isin(specs))].set_index("specification").reindex(specs)
        for k, (s, r) in enumerate(t.iterrows()):
            col = MAIN if k == 0 else ALT
            ax.plot([r["ci_low"], r["ci_high"]], [k, k], color=col, linewidth=2, solid_capstyle="round")
            ax.plot(r["effect"], k, "o", color=col, markersize=6, markeredgecolor="white")
        pr = perm[perm["treatment"] == treat].iloc[0]
        k = len(specs)
        ax.fill_betweenx([k - 0.25, k + 0.25], pr["null_p2_5"], pr["null_p97_5"], color=GRID)
        ax.plot(pr["null_mean"], k, "|", color=INK, markersize=10)
        ax.text(pr["null_p97_5"], k, f"  centre {pr['null_mean']:+.2f}", va="center", fontsize=7.5, color=MUTED)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(range(len(short))); ax.set_yticklabels(short)
        lab, unit, per = TREATMENTS[treat]
        ax.set_title(f"{treat}  {lab}", fontsize=9.5, loc="left")
        ax.set_xlabel(f"Change in slum share (pp)\nper +{per} {unit}, 95% CI")
        ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Double machine learning: effect on urban slum share (11.1.1), country-clustered 95% CIs",
                 x=0.06, ha="left", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "panel_wide.csv", int(sys.argv[2]) if len(sys.argv) > 2 else 10)
