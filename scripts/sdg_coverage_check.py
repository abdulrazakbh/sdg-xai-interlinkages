"""
SDG infrastructure-interlinkage paper - Step 1: data download and coverage check
================================================================================

What this script does
---------------------
1. Downloads every indicator in the research plan from the UN SDG Global
   Database API (https://unstats.un.org/SDGAPI), all countries, all years.
2. Keeps country-level records only (drops regional/world aggregates).
3. Reduces each series to one "headline" value per country-year
   (totals across sex/age/location; if a series has no total, the
   disaggregation with the best coverage is kept and recorded).
4. Picks the headline series for each indicator (best coverage).
5. Downloads the controls from the World Bank API
   (GDP per capita, urbanisation, population, income group).
6. Writes:
     outputs/series_coverage.csv     coverage of every series
     outputs/indicator_coverage.csv  coverage of the headline series + keep/drop flag
     outputs/coverage_heatmap.png    % of countries reporting, indicator x year
     outputs/india_coverage.csv      years of data India has per indicator
     outputs/panel_long.csv          country-year-indicator values (headline series)
     outputs/panel_wide.csv          country-year panel ready for modelling
     outputs/run_log.txt             log of the run

How to run
----------
    pip install requests pandas numpy matplotlib pycountry
    python sdg_coverage_check.py

Raw API downloads are cached in data/raw/, so a second run is fast.
Delete that folder to force a fresh download.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
import requests

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

YEAR_START, YEAR_END = 2010, 2023
KEEP_THRESHOLD = 0.60          # minimum gap-filled country-year coverage to keep
INTERP_LIMIT = 3               # max consecutive missing years filled by interpolation
PAGE_SIZE = 5000

UN_API = "https://unstats.un.org/SDGAPI/v1/sdg/Indicator/Data"
WB_API = "https://api.worldbank.org/v2"

# code: (short name, role, direction)  direction +1 = higher is better, -1 = lower is better
INDICATORS: dict[str, tuple[str, str, int]] = {
    "9.1.1":  ("Rural pop. within 2 km of all-season road", "target_sdg9", +1),
    "9.1.2":  ("Passenger and freight volumes",              "target_sdg9", +1),
    "9.2.1":  ("Manufacturing value added",                  "target_sdg9", +1),
    "9.4.1":  ("CO2 per unit of value added",                "target_sdg9", -1),
    "9.5.1":  ("R&D expenditure (% GDP)",                    "target_sdg9", +1),
    "9.5.2":  ("Researchers per million",                    "target_sdg9", +1),
    "9.c.1":  ("Mobile network coverage",                    "target_sdg9", +1),
    "11.1.1": ("Urban population in slums",                  "target_sdg11", -1),
    "11.2.1": ("Convenient access to public transport",      "target_sdg11", +1),
    "11.3.1": ("Land consumption / population growth",       "target_sdg11", -1),
    "11.6.1": ("Municipal solid waste collected & managed",  "target_sdg11", +1),
    "11.6.2": ("Urban PM2.5",                                "target_sdg11", -1),
    "6.1.1":  ("Safely managed drinking water",              "driver", +1),
    "6.2.1":  ("Safely managed sanitation",                  "driver", +1),
    "6.4.1":  ("Water-use efficiency",                       "driver", +1),
    "6.4.2":  ("Water stress",                               "driver", -1),
    "7.1.1":  ("Access to electricity",                      "driver", +1),
    "7.1.2":  ("Clean cooking fuels",                        "driver", +1),
    "7.2.1":  ("Renewable share of final energy",            "driver", +1),
    "7.3.1":  ("Energy intensity",                           "driver", -1),
    "3.6.1":  ("Road traffic death rate",                    "driver", -1),
    "12.5.1": ("National recycling rate",                    "driver", +1),
    "17.6.1": ("Fixed broadband subscriptions",              "driver", +1),
    "17.8.1": ("Internet users",                             "driver", +1),
    "1.1.1":  ("Population below intl. poverty line",        "control", -1),
}

# Optional: force a specific series for an indicator, e.g. {"9.2.1": "NV_IND_MANFPC"}
PREFERRED_SERIES: dict[str, str] = {
    "9.4.1": "EN_ATM_CO2GDP",   # CO2 per unit of GDP (not total emissions)
    "6.2.1": "SH_SAN_SAFE",     # safely managed sanitation (not open defecation)
    "9.c.1": "IT_MOB_3GNTWK",   # 3G coverage (2G is saturated)
    "9.2.1": "NV_IND_MANF",     # MVA share of GDP (per capita would mirror the GDP control)
}

WB_CONTROLS = {
    "NY.GDP.PCAP.KD": "gdp_pc_const2015usd",
    "SP.URB.TOTL.IN.ZS": "urban_pct",
    "SP.POP.TOTL": "population",
}

# Dimension codes that mean "total / all" in the UN SDG database
TOTAL_CODES = {
    "ALLAREA", "BOTHSEX", "ALLAGE", "_T", "ALL", "TOTAL", "_Z", "_X",
    "ALLSEC", "ALLMODE", "NOSPECIFIC",
}
IGNORE_DIMS = {"Reporting Type", "Nature", "Units"}

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[logging.FileHandler(OUT / "run_log.txt", mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sdg")

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "sdg-research-script"})


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def get_json(url: str, params: dict, retries: int = 4) -> dict | list:
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=180)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            wait = 5 * attempt
            log.warning("Request failed (%s), attempt %d/%d, retrying in %ds",
                        exc, attempt, retries, wait)
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url} {params}")


def country_codes() -> dict[str, str]:
    """M49 numeric code (as UN API string, no leading zeros) -> ISO3."""
    return {str(int(c.numeric)): c.alpha_3 for c in pycountry.countries}


def to_number(v) -> float:
    if v is None:
        return np.nan
    s = str(v).strip().replace(",", "")
    for ch in "<>~≤≥":
        s = s.replace(ch, "")
    try:
        return float(s)
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------------
# 1. Download from the UN SDG API
# ----------------------------------------------------------------------------

def download_indicator(code: str) -> pd.DataFrame:
    cache = RAW / f"un_{code.replace('.', '_')}.csv"
    if cache.exists():
        log.info("%-7s cached", code)
        return pd.read_csv(cache, dtype={"geoAreaCode": str})

    rows, page, total_pages = [], 1, 1
    while page <= total_pages:
        js = get_json(UN_API, {"indicator": code, "page": page, "pageSize": PAGE_SIZE})
        total_pages = js.get("totalPages", 1) or 1
        for rec in js.get("data", []):
            dims = rec.get("dimensions") or {}
            attrs = rec.get("attributes") or {}
            rows.append({
                "indicator": code,
                "series": rec.get("series"),
                "seriesDescription": rec.get("seriesDescription"),
                "geoAreaCode": str(rec.get("geoAreaCode")),
                "geoAreaName": rec.get("geoAreaName"),
                "year": rec.get("timePeriodStart"),
                "value": rec.get("value"),
                "nature": attrs.get("Nature"),
                "units": attrs.get("Units"),
                "reporting_type": dims.get("Reporting Type"),
                "dims": json.dumps({k: v for k, v in dims.items() if k not in IGNORE_DIMS},
                                   sort_keys=True),
            })
        log.info("%-7s page %d/%d", code, page, total_pages)
        page += 1
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    log.info("%-7s %d records downloaded", code, len(df))
    return df


# ----------------------------------------------------------------------------
# 2-4. Clean, choose headline values and series
# ----------------------------------------------------------------------------

def headline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One value per series x country x year."""
    out = []
    for series, g in df.groupby("series"):
        # Prefer global (harmonised) reporting over national, per country-year,
        # so countries with only national data are not lost
        rank = g["reporting_type"].map({"G": 0, "N": 1}).fillna(2)
        best_rank = rank.groupby([g["iso3"], g["year"]]).transform("min")
        g = g[rank == best_rank]

        def is_total(dstr: str) -> bool:
            d = json.loads(dstr)
            return all(v in TOTAL_CODES for v in d.values())

        tot = g[g["dims"].map(is_total)]
        if not tot.empty:
            g = tot
            chosen = "total"
        else:
            # No total available: keep the disaggregation with the best coverage
            counts = g.dropna(subset=["value_num"]).groupby("dims").size()
            best = counts.idxmax() if not counts.empty else g["dims"].iloc[0]
            g = g[g["dims"] == best]
            chosen = best
        g = (g.dropna(subset=["value_num"])
               .groupby(["series", "seriesDescription", "iso3", "year"], as_index=False)
               ["value_num"].mean())
        g["disaggregation_used"] = chosen
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def coverage_stats(g: pd.DataFrame, n_countries: int, years: list[int]) -> dict:
    n_years = len(years)
    wide = (g.pivot_table(index="iso3", columns="year", values="value_num")
              .reindex(columns=years))
    raw_obs = int(wide.notna().sum().sum())
    filled = wide.interpolate(axis=1, limit=INTERP_LIMIT, limit_area="inside")
    filled_obs = int(filled.notna().sum().sum())
    per_country = wide.notna().sum(axis=1)
    return {
        "countries_with_data": int((per_country > 0).sum()),
        "raw_obs": raw_obs,
        "raw_coverage": raw_obs / (n_countries * n_years),
        "gapfilled_coverage": filled_obs / (n_countries * n_years),
        "median_years_per_country": float(per_country[per_country > 0].median())
        if (per_country > 0).any() else 0.0,
        "first_year": int(g["year"].min()) if not g.empty else None,
        "latest_year": int(g["year"].max()) if not g.empty else None,
    }


# ----------------------------------------------------------------------------
# 5. World Bank controls
# ----------------------------------------------------------------------------

def download_wb() -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_ctrl, cache_meta = RAW / "wb_controls.csv", RAW / "wb_countries.csv"
    if cache_ctrl.exists() and cache_meta.exists():
        log.info("World Bank data cached")
        return pd.read_csv(cache_ctrl), pd.read_csv(cache_meta)

    frames = []
    for wb_code, name in WB_CONTROLS.items():
        js = get_json(f"{WB_API}/country/all/indicator/{wb_code}",
                      {"format": "json", "date": f"{YEAR_START}:{YEAR_END}", "per_page": 20000})
        recs = js[1] if isinstance(js, list) and len(js) > 1 and js[1] else []
        frames.append(pd.DataFrame({
            "iso3": [r["countryiso3code"] for r in recs],
            "year": [int(r["date"]) for r in recs],
            name: [r["value"] for r in recs],
        }).set_index(["iso3", "year"]))
        log.info("World Bank %s: %d records", wb_code, len(recs))
    ctrl = pd.concat(frames, axis=1).reset_index()

    js = get_json(f"{WB_API}/country", {"format": "json", "per_page": 400})
    meta = pd.DataFrame([{
        "iso3": c["id"],
        "income_group": c["incomeLevel"]["value"],
        "region": c["region"]["value"],
    } for c in js[1] if c["region"]["value"] != "Aggregates"])

    ctrl.to_csv(cache_ctrl, index=False)
    meta.to_csv(cache_meta, index=False)
    return ctrl, meta


# ----------------------------------------------------------------------------
# 6. Outputs
# ----------------------------------------------------------------------------

def plot_heatmap(panel_long: pd.DataFrame, n_countries: int, years: list[int], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    share = (panel_long.groupby(["indicator", "year"])["iso3"].nunique()
                       .unstack().reindex(columns=years) / n_countries * 100)
    order = [c for c in INDICATORS if c in share.index]
    share = share.reindex(order).fillna(0)
    labels = [f"{c}  {INDICATORS[c][0]}" for c in order]

    fig, ax = plt.subplots(figsize=(12, 0.38 * len(order) + 1.8))
    im = ax.imshow(share.values, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=90, fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Year")
    ax.set_title(f"Share of countries reporting each indicator (n = {n_countries}), "
                 f"headline series", fontsize=10)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("% of countries with data")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    years = list(range(YEAR_START, YEAR_END + 1))
    m49_to_iso3 = country_codes()

    ctrl, meta = download_wb()
    # Country universe = World Bank economies that are also ISO countries
    universe = sorted(set(meta["iso3"]) & set(m49_to_iso3.values()))
    n_countries = len(universe)
    log.info("Country universe: %d countries", n_countries)

    series_rows, panel_parts = [], []
    for code, (name, role, direction) in INDICATORS.items():
        try:
            df = download_indicator(code)
        except RuntimeError as exc:
            log.error("%-7s download failed: %s", code, exc)
            continue
        if df.empty:
            log.warning("%-7s returned no data", code)
            continue

        df["iso3"] = df["geoAreaCode"].astype(str).str.lstrip("0").map(m49_to_iso3)
        df = df[df["iso3"].isin(universe)].copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df[df["year"].between(YEAR_START, YEAR_END)]
        df["value_num"] = df["value"].map(to_number)

        hl = headline_rows(df)
        if hl.empty:
            log.warning("%-7s no usable country-level values", code)
            continue
        hl["year"] = hl["year"].astype(int)

        for series, g in hl.groupby("series"):
            stats = coverage_stats(g, n_countries, years)
            series_rows.append({
                "indicator": code, "indicator_name": name, "role": role,
                "direction": direction, "series": series,
                "series_description": g["seriesDescription"].iloc[0],
                "disaggregation_used": g["disaggregation_used"].iloc[0],
                **stats,
            })

        cand = [r for r in series_rows if r["indicator"] == code]
        best = PREFERRED_SERIES.get(code) or max(cand, key=lambda r: r["gapfilled_coverage"])["series"]
        for r in cand:
            r["headline"] = r["series"] == best
        part = hl[hl["series"] == best].assign(indicator=code)
        panel_parts.append(part)
        log.info("%-7s headline series %s", code, best)

    if not panel_parts:
        log.error("No indicator data downloaded. Check your internet connection.")
        return

    series_cov = pd.DataFrame(series_rows)
    series_cov.to_csv(OUT / "series_coverage.csv", index=False)

    ind_cov = series_cov[series_cov["headline"]].copy()
    ind_cov["keep"] = ind_cov["gapfilled_coverage"] >= KEEP_THRESHOLD
    ind_cov["decision"] = np.where(ind_cov["keep"], "keep (main model)",
                                   "sensitivity run only")
    missing = [c for c in INDICATORS if c not in set(ind_cov["indicator"])]
    if missing:
        ind_cov = pd.concat([ind_cov, pd.DataFrame([{
            "indicator": c, "indicator_name": INDICATORS[c][0], "role": INDICATORS[c][1],
            "direction": INDICATORS[c][2], "decision": "no data returned", "keep": False,
        } for c in missing])], ignore_index=True)
    ind_cov = ind_cov.sort_values(["keep", "gapfilled_coverage"], ascending=[False, False])
    ind_cov.to_csv(OUT / "indicator_coverage.csv", index=False)

    panel_long = pd.concat(panel_parts, ignore_index=True)
    panel_long = panel_long[["iso3", "year", "indicator", "series", "value_num"]]
    panel_long.to_csv(OUT / "panel_long.csv", index=False)

    wide = (panel_long.pivot_table(index=["iso3", "year"], columns="indicator",
                                   values="value_num").reset_index())
    grid = pd.MultiIndex.from_product([universe, years], names=["iso3", "year"]).to_frame(index=False)
    wide = (grid.merge(wide, on=["iso3", "year"], how="left")
                .merge(ctrl, on=["iso3", "year"], how="left")
                .merge(meta, on="iso3", how="left"))
    wide.insert(1, "country", wide["iso3"].map(
        lambda a: getattr(pycountry.countries.get(alpha_3=a), "name", a)))
    wide.to_csv(OUT / "panel_wide.csv", index=False)

    india = (panel_long[panel_long["iso3"] == "IND"].groupby("indicator")["year"]
             .agg(years_with_data="nunique", first_year="min", latest_year="max")
             .reindex(list(INDICATORS)).reset_index())
    india.insert(1, "indicator_name", india["indicator"].map(lambda c: INDICATORS[c][0]))
    india.to_csv(OUT / "india_coverage.csv", index=False)

    plot_heatmap(panel_long, n_countries, years, OUT / "coverage_heatmap.png")

    kept = int(ind_cov["keep"].sum())
    log.info("Done. %d of %d indicators pass the %.0f%% gap-filled coverage threshold.",
             kept, len(INDICATORS), KEEP_THRESHOLD * 100)
    log.info("Outputs written to %s", OUT)


if __name__ == "__main__":
    main()
