# Physics-guided learning for node-availability prediction in solar-powered IoT sensor networks

Code, data and results for:

> I. Dagal, A. L. Mahamat, *Physics-Guided Learning for Energy Autonomy and Data Availability Prediction in Solar-Powered IoT Sensor Networks* (manuscript submitted, 2026).

A daily energy-balance model of a VRLA-buffered solar node (low-voltage disconnect, reconnection threshold, cold deep-discharge latch, per-node harvest multiplier) is assimilated with the reported availability and used as the prior of a LightGBM residual learner with monotone physical constraints. The study uses one year (13 June 2025 – 13 June 2026) of telemetry from a 20-node air-quality network in Bucharest.

## Requirements
Python ≥ 3.10. `requirements.txt` lists the exact versions used (they need Python ≥ 3.11); on Python 3.10, install the same packages without versions: `pip install numpy pandas scipy pvlib scikit-learn lightgbm matplotlib` (checked: results reproduce).

    pip install -r requirements.txt

## Reproducing the results
Run from the repository root, in this order:

| Step | Script | Output |
|---|---|---|
| 1 | `drivers.py` | `drivers.csv` – daily clear-sky in-plane irradiation, clearness and temperature drivers |
| 2 | `select_structure.py` | `structure_selection.csv` – physics-model structure, selected on the training period only |
| 3 | `exp2_temporal.py` | `res2_temporal.csv` – chronological and rolling-origin experiments, horizons 1–28 d (Table 2) |
| 4 | `boot.py` | `boot_temporal.csv`, `boot_spatial.csv` – weekly block-bootstrap confidence intervals (Table 3) |
| 5 | `exp2_spatial.py`, `exp2_spatial_b.py` | `res2_spatial.csv`, `res2_spatial_all.csv` – leave-one-node-out designs (Table 4) |
| 6 | `repres2.py` | `res_repres.csv` – availability-weighted PM2.5 representativeness (Table 5) |
| 7 | `fig1.py`, `fig_data.py`, `fig_temporal.py`, `fig_spatial.py`, `fig_repres.py` | `figs/*.png` |
| 8 | `tables.py` | manuscript tables (Markdown) |

The result files of every step are included, so each step can also be checked on its own. Steps 1–3 were re-run from this repository and reproduce `drivers.csv`, `structure_selection.csv` and `res2_temporal.csv` exactly (step 3 takes about 4 min on a laptop CPU). Steps 3 and 5 also write per-prediction files (`p2_*.csv`, `pred_*.csv`), which are not included.

## Modules
- `core.py` – data loading, commissioning mask, common-mode outage days, original physics
- `phys2.py` – VRLA daily energy balance with LVD, reconnection threshold, cold latch and per-node harvest multiplier
- `feats2.py` – causal assimilation of reported availability, h-day propagation, data and physics features
- `models.py` – data-driven learner, physics prior, physics-prior residual boosting with monotone constraints, metrics (Brier score, AUROC, network-availability MAE)
- `data/D_code/leadacid.py` – temperature-dependent lead-acid battery model used by the physics layer
- `style.py` – figure style

## Data
See `data/README.md`. The data path can be changed with `export PGL_DATA=/path/to/data`.

## Licence
Code: MIT licence (`LICENSE`). Data: CC BY 4.0 (`data/README.md`).
