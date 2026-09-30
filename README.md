# TT-vs-EO-ML: Leakage-free machine learning pipeline for testicular torsion vs epididymo-orchitis

Analysis code accompanying the manuscript:

> Çift A, Kutlu H, et al. *Predicting Testicular Torsion in Patients with Acute Scrotum Using Hybrid Machine Learning and Explainable Artificial Intelligence: A Web-Based Clinical Decision Support System.* (submitted, 2026)

The pipeline distinguishes testicular torsion (TT) from epididymo-orchitis (EO) using age and 16 complete-blood-count-derived features, with an emphasis on avoiding the optimism that affects much of the clinical ML literature:

* a fixed, stratified hold-out set that is never resampled, scaled or used for selection;
* repeated **nested cross-validation** (5 × 2 outer, 5 inner) with Bayesian hyper-parameter search (Optuna);
* class-imbalance strategy (none / ROS / SMOTE / Borderline-SMOTE / SVMSMOTE / ADASYN / SMOTE-ENN / SMOTE-Tomek) treated as a **tunable hyper-parameter inside the inner loop**, applied to training folds only;
* hybrid binary **PSO-GWO** feature selection benchmarked against 11 alternatives with a resampling stability analysis;
* an **age-only logistic regression reference** to quantify the incremental value of hematological markers;
* explainability with SHAP, LIME, permutation importance, model-based importance and PDP/ICE, plus rank-agreement statistics;
* a four-feature logistic **nomogram**, a four-item bedside score and a two-module **Dash web application**.

## Repository layout

| File | Purpose |
|---|---|
| `config.py` | paths, feature sets, experiment settings (edit `DATA_PATH` / `OUT_DIR`) |
| `utils.py` | metrics, DeLong test, bootstrap CIs, effect sizes, decision curves |
| `00_prepare_data.py` | cleaning, consistency checks, leakage audit, fixed stratified split |
| `01_descriptive_univariate.py` | descriptive statistics, univariate tests (FDR), univariate ROC |
| `02_nested_cv_models.py` | nested CV for 10 classifiers, resampling comparison, hold-out evaluation, calibration, DCA |
| `03_xai.py` | SHAP / LIME / permutation / model-based importance, PDP-ICE, rank agreement |
| `04_feature_selection.py` | hybrid PSO-GWO and comparator selectors, stability analysis |
| `05_nomogram.py` | logistic nomogram, calibration, simple bedside score |
| `06_webapp.py` | Dash clinical decision-support application |
| `run_all.bat` | runs the full pipeline (Windows) |

## Reproducing the analysis

```bash
conda create -n tteo python=3.12 -y
conda activate tteo
pip install -r requirements.txt
```

Place the dataset (`data.xlsx`, one sheet named `Sayfa1`) at the path given by `DATA_PATH` in `config.py`, then:

```bash
python 00_prepare_data.py
python 01_descriptive_univariate.py
python 04_feature_selection.py
python 02_nested_cv_models.py --features-json outputs/04_feature_selection/selected_features.json --fs-method PSO-GWO
python 03_xai.py --tag _fs_psogwo
python 05_nomogram.py --features Age PDW MPV NLR --tag _reduced
python 06_webapp.py            # http://127.0.0.1:8050
```

Sensitivity analyses reported in the paper:

```bash
python 02_nested_cv_models.py --drop-age   --clf logreg rf et gb xgb cat
python 02_nested_cv_models.py --age-max 35 --clf logreg rf et gb xgb cat
```

All random seeds are fixed (`RANDOM_STATE = 42`). Full runtime on an 8-core workstation is approximately 4–5 hours; set `QUICK=1` in the environment for a smoke test (a few minutes).

## Data availability

The patient-level dataset is not distributed with this repository because it contains clinical data; it is available from the corresponding author on reasonable request and subject to ethics-committee approval. The expected input format is documented in `config.py` (`RENAME_MAP`).

## Software versions used for the reported results

Python 3.12; scikit-learn 1.9.1; imbalanced-learn 0.14.2; XGBoost 3.4.1; LightGBM 4.7.0; CatBoost 1.2.10; Optuna 5.0.0; SHAP 0.52.0; LIME 0.2.0.1; statsmodels 0.15.0; Dash 4.4.1.

## Citation

If you use this code, please cite the manuscript above and the archived release (Zenodo DOI in `CITATION.cff`).

## License

MIT License — see `LICENSE`.
