# PACE PCC ML — Reproducible, one-command pipeline

This repository contains code to predict **Phytoplankton Community Composition (PCC)** from hyperspectral sea-surface reflectance (Rrs) and optional ancillary features using **XGBoost** with a **multi-output** head. Targets (7): Diatoms (dia), Chlorophytes (chl), Cyanobacteria (cya), Coccolithophores (coc), Dinoflagellates (din), Phaeocystis (pha), and total chlorophyll-a (tot\_chla).
**Hyperparameter optimization:** Optuna.
**Explainability:** SHAP (summary plots per target).

---

## Before you install

> You can use either **Conda** or **vanilla Python + venv**. Pick one.

### Option A — Conda (recommended for exact reproducibility)

1. **Create / update the environment**

   ```bash
   conda env create -f environment.yml            # first time
   # or, if you've updated environment.yml:
   conda env update -f environment.yml --prune
   conda activate phyx
   ```
2. **(Optional) Developer tools**

   * Quick tests: `pytest` (add to `environment.yml` or `pip install pytest`)
   * Git hooks: `pre-commit` (optional)

     ```bash
     pip install pre-commit
     pre-commit install
     ```

### Option B — Plain Python + venv (no Conda)

1. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate         # Windows: .venv\Scripts\activate
   ```

2. **Install pinned requirements**

   ```bash
   pip install -r requirements.txt
   # if you need Parquet support and it's not included:
   pip install pyarrow
   ```

---

## Install the package (both options)

From the **repo root** (folder with `pyproject.toml` and `src/`):

```bash
pip install -e . --no-deps
phyx --help
# if PATH scripts aren’t picked up:
python -m phyx.cli --help
```

---

## Data expectations

* The pipeline is **dataset-agnostic**: it uses whatever feature columns are present in `df_rrs.pqt` (and optional `df_env.pqt`) without internal filtering.
* Requirements:

  * **Train and test must have identical feature schemas** (same column names *and* order). Targets live only in `df_phyto.pqt`.
  * All feature/target columns must be **numeric** (no NaN/inf after your own preprocessing).
  * Keep units and scaling consistent across files.
* Default file layout (Parquet):

  ```
  data_any_dataset/
    df_rrs.pqt      # features (e.g., Rrs_* and/or ancillary columns)
    df_phy.pqt      # targets: dia, chl, cya, coc, din, pha, tot_chla
    df_env.pqt      # optional ancillary features (same rows), optional
  ```
* If you choose to restrict wavelengths (e.g., to a sensor band range), do that upstream in your **preprocessing** and save as a separate dataset folder.

---

## One-command pipeline

The `run-all` command:

1. Runs **Optuna HPO** on the training split (KFold inside objective),
2. **Retrains** the model on the **full training split** using best params,
3. **Evaluates** on the **held-out test split** (metrics.json),
4. Computes **SHAP** values + **summary plots** per target on the test set,
5. Optionally fits a final **production model** on **all data** (train+test) after evaluation.

```bash
phyx run-all \
  --data-path data_directory \
  --n-trials 50 \      # number of HPO trials
  --cv-folds 5 \       # optional, default=3
  --outdir artifacts/pace_run \
  --fit-all            # optional for production model: also saves model_production.pkl (trained on ALL data - train + test)
```

**Outputs (in `--outdir`)**

* `best_params.json` — Optuna best hyperparameters
* `model.pkl` — trained on the **training split** with best params
* `metrics.json` — MSE / RMSE / MAE / R² on the **test split**
* `shap_values.npz` and `shap_summary_target_*.png` — XAI artifacts from the test split
* `model_production.pkl` — *(only if `--fit-all`)* trained on **train+test** (no metrics)

> 🔁 Want just the tuning step?
>
> ```bash
> phyx hpo --data-path data_any_dataset --n-trials 50 --outdir artifacts/hpo
> ```

---

## Training recommendations

* **Report** metrics from the model trained on the training split and evaluated on the held-out **test split**. Keep the test set “clean.”
* **Deploy** a final **production** model trained on **all available data** (train+test) **after** you’ve validated. Save it separately (e.g., `model_production.pkl`) and **do not** report its metrics as if they came from a clean test.

---

## Quick sanity run (fast)

If you want a quick check without burning compute:

```bash
phyx run-all \
  --data-path data_any_dataset \
  --n-trials 5 \
  --cv-folds 2 \       # quick test
  --outdir artifacts/smoke
```

---

## Troubleshooting

* **`phyx: command not found`**
  Ensure editable install succeeded: `pip install -e . --no-deps`.
  Try `python -m phyx.cli --help`.

* **Parquet read error / `pyarrow` not found**
  Install `pyarrow` (Conda: add to `environment.yml`; venv: `pip install pyarrow`).

* **Import errors in tests**
  Make sure you’ve run `pip install -e .` so `phyx` is importable.

---

## What’s inside (high level)

```
src/phyx/
  pipeline/
    data_loader.py       # load X (Rrs), optional env, and Y (targets)
    optuna_hpo.py        # Optuna objective (KFold CV inside trials)
    model_trainer.py     # XGB multi-output training & saving
    model_evaluator.py   # metrics (MSE/RMSE/MAE/R²)
  synthesis/
    shap_runner.py       # SHAP values + per-target summary plots
```

> Minimal refactor goal: keep modules close to the notebook & existing pipeline, favor clarity over framework ceremony. SWEs can further modularize as needed.
