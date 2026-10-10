"""
Block-permutation placebo for the DML slum estimates.

Each draw reassigns every country's full treatment series to another, randomly chosen
country (same years), so each placebo series keeps its own persistence over time and
the shared global trend, but loses any link to the recipient's slum share.
  * Null centre  = bias the estimator picks up from the shared trend alone.
  * Null spread  = sampling uncertainty under the real serial-correlation structure.
Reported p-value = share of placebo draws at least as far from the null centre as the
observed estimate (two-sided), with the +1 correction.

Run after dml_slums.py:  python dml_block_placebo.py path/to/panel_wide.csv [draws]
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dml", HERE / "dml_slums.py")
dml = importlib.util.module_from_spec(spec); spec.loader.exec_module(dml)


def main(panel, B):
    lev = dml.v2.load_levels(Path(panel))
    res = pd.read_csv(HERE / "results_dml" / "dml_results.csv")
    rows = []
    rng = np.random.default_rng(2026)
    for treat, (lab, unit, per) in dml.TREATMENTS.items():
        X_cols = dml.controls_for(treat)
        d = lev.dropna(subset=[dml.OUTCOME, treat, "log_gdp_pc"]).reset_index(drop=True)
        obs = res[(res.treatment == treat) & (res.specification == "Cross-country, XGBoost nuisance")]["effect"].iloc[0]
        series = d.set_index(["iso3", "year"])[treat]
        countries = d["iso3"].unique()
        draws = []
        for b in range(B):
            donor = dict(zip(countries, rng.permutation(countries)))
            Dp = np.array([series.get((donor[c], yr), np.nan) for c, yr in zip(d["iso3"], d["year"])])
            ok = ~np.isnan(Dp)
            r = dml.dml_plr(d.loc[ok, dml.OUTCOME].values, Dp[ok], d.loc[ok, X_cols].values,
                            d.loc[ok, "iso3"].values, "XGBoost", R=1, seed=3000 + b)
            draws.append(r["theta"] * per)
        draws = np.array(draws)
        c = draws.mean()
        p = (np.sum(np.abs(draws - c) >= abs(obs - c)) + 1) / (B + 1)
        rows.append({"treatment": treat, "label": lab, "observed_effect": obs, "null_mean": c,
                     "null_p2_5": np.quantile(draws, 0.025), "null_p97_5": np.quantile(draws, 0.975),
                     "bias_corrected_effect": obs - c, "permutation_p": p, "draws": B})
        print(f"{treat:<7} observed {obs:+.2f}  null centre {c:+.2f}  null 95% [{rows[-1]['null_p2_5']:+.2f}, "
              f"{rows[-1]['null_p97_5']:+.2f}]  block-permutation p={p:.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results_dml" / "dml_block_permutation_placebo.csv", index=False)
    dml.plot(res, out, HERE / "results_dml" / "dml_effects.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "panel_wide.csv", int(sys.argv[2]) if len(sys.argv) > 2 else 50)
