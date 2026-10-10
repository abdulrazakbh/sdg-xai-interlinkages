# Importance is not impact: SDG interlinkage analysis

Code and processed data for the manuscript **"Importance is not impact: explainable and double machine learning evidence on infrastructure-related SDG interlinkages"** (B. H. Abdul Razak et al., submitted to *Sustainable Development*).

The analysis moves through three steps:

1. **Correlation.** Pooled Spearman correlations between SDG drivers and outcomes (the conventional rule).
2. **Conditional association.** XGBoost level and five-year change models, validated on countries held out of training and explained with SHAP, with 100 country-bootstrap resamples.
3. **Effect.** Double machine learning for the three leading drivers of urban slum share (SDG 11.1.1).

Coverage: up to 209 countries, 2010–2023.

## Repository contents

| Folder | Contents |
| --- | --- |
| `scripts/` | The four analysis scripts, in run order (below) |
| `data/processed/` | Country-year panel used in the paper (`panel_wide.csv`), long-format indicator values, and coverage tables |
| `results/xai/` | Model performance, SHAP importance with bootstrap intervals, change-model results, correlation benchmark, income-group and India results |
| `results/dml/` | Double machine learning estimates and placebo results |
| `figures/` | Figures 1–6 and S1–S2 as in the manuscript (300 dpi) |

Raw API downloads are not stored here. Step 1 below re-creates them in `scripts/data/raw/`.

## Requirements

Python 3.10 or later. Install the packages with:

```
pip install -r requirements.txt
```

The results were produced with Python 3.13 and the versions pinned in `requirements.txt`. Newer versions should give the same conclusions, though decimals may differ slightly.

## How to reproduce

Run from inside `scripts/`:

```
cd scripts

# 1. Download the UN SDG and World Bank data, check coverage, build the panel (about 45-90 min, needs internet)
python sdg_coverage_check.py

# 2. Level and change models, SHAP, 100 bootstrap resamples (about 15 min)
python sdg_xai_model_v2.py outputs/panel_wide.csv 100

# 3. Double machine learning for the slum drivers, 10 repeated splits (about 20 min)
python dml_slums.py outputs/panel_wide.csv 10

# 4. Block-permutation placebo (50 draws) and Figure 6 (about 10 min)
python dml_block_placebo.py outputs/panel_wide.csv 50
```

To skip the download, replace `outputs/panel_wide.csv` with `../data/processed/panel_wide.csv` in steps 2–4. Outputs are written to `scripts/outputs/`, `scripts/results_v2/` and `scripts/results_dml/`.

The UN and World Bank databases are revised over time, so a fresh download may differ slightly from the panel in `data/processed/`. That panel was downloaded on 9 October 2026 and is the exact version used in the paper.

## Data sources

- United Nations Statistics Division, SDG Global Database API: https://unstats.un.org/SDGAPI/v1/sdg/ (accessed 9 October 2026)
- World Bank, World Development Indicators: https://datatopics.worldbank.org/world-development-indicators/ (accessed 9 October 2026; CC BY 4.0)

The processed data in this repository are derived from these sources and are shared with attribution under the sources' terms.

## Licence

Code: MIT License (see `LICENSE`). Processed data and figures: CC BY 4.0.

## Citation

If you use this code or data, please cite the paper (details to be added on publication) and this repository (see `CITATION.cff`).

## Contact

B. H. Abdul Razak, Department of Civil Engineering, JSS Academy of Technical Education, Bengaluru, India. abdulrazak.b.h@gmail.com
