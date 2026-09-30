# -*- coding: utf-8 -*-
"""
ADIM 2 — Nested cross-validation ile optimize edilmiş sınıflandırıcılar.

Tasarım (sızıntısız):
  • Sabit tabakalı hold-out (00 adımında): test seti %100 orijinal, hiçbir zaman yeniden örneklenmez.
  • EĞİTİM seti üzerinde nested CV:
      dış döngü  : RepeatedStratifiedKFold(OUTER_FOLDS × OUTER_REPEATS) → yansız performans tahmini
      iç döngü   : StratifiedKFold(INNER_FOLDS) + Optuna → hiperparametre + ÖRNEKLEYİCİ seçimi
    Örnekleyici (none / ROS / SMOTE / Borderline / SVMSMOTE / ADASYN / SMOTEENN / SMOTETomek) bir
    hiperparametre olarak iç döngüde seçilir; imblearn Pipeline sayesinde yeniden örnekleme yalnızca
    iç eğitim katına uygulanır, doğrulama/test katları orijinal kalır.
  • Ölçekleme de pipeline içinde (fold-içi fit) → test bilgisinin sızması yok.
  • Nihai model: tüm eğitim seti üzerinde iç döngü tekrar çalıştırılır, en iyi konfigürasyon fit edilir,
    hold-out test seti üzerinde bootstrap %95 GA ile raporlanır.
  • Karar eşiği: 0.5 (birincil) + eğitim OOF tahminlerinden Youden eşiği (ikincil, teste taşınır).

Kullanım:
  python 02_nested_cv_models.py                     # tüm sınıflandırıcılar, FEATURE_SET=no_percent
  python 02_nested_cv_models.py --clf rf xgb cat    # seçili sınıflandırıcılar
  python 02_nested_cv_models.py --drop-age          # yaş hariç duyarlılık analizi (ayrı klasöre yazar)
  set QUICK=1 && python 02_nested_cv_models.py      # hızlı duman testi

Çıktılar (OUT_DIR/02_models[ _noage ]/):
  nested_cv_fold_results.csv       : her dış kat için tüm metrikler + seçilen hiperparametreler/örnekleyici
  Table4_nested_cv_summary.xlsx    : model başına mean ± SD ve %95 GA
  Table5_holdout_test.xlsx         : hold-out test metrikleri + bootstrap GA (0.5 ve Youden eşiği)
  Table6_delong_vs_best.xlsx       : en iyi modele karşı DeLong testi
  Table7_sampler_comparison.xlsx   : en iyi model için her örnekleyicinin ayrı ayrı CV performansı
  best_configs.json, final_models/<clf>.joblib, test_predictions.csv, oof_predictions.csv
  Fig_ROC_test.png, Fig_PR_test.png, Fig_calibration.png, Fig_DCA.png, Fig_confusion_matrices.png,
  Fig_nested_cv_boxplot.png, Fig_sampler_comparison.png
"""
import argparse, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
import config as C
from utils import binary_metrics, bootstrap_metrics, delong_test, delong_ci, youden_threshold, decision_curve, fmt_ci

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# ------------------------------------------------------------------ Argümanlar
ap = argparse.ArgumentParser()
ap.add_argument("--clf", nargs="*", default=C.CLASSIFIERS)
ap.add_argument("--drop-age", action="store_true")
ap.add_argument("--age-max", type=float, default=None, help="yalnız Age<=N hastalar (yaş-örtüşen alt kohort duyarlılık analizi)")
ap.add_argument("--feature-set", default=C.FEATURE_SET)
ap.add_argument("--features-json", default=None, help="04 adımının selected_features.json yolu")
ap.add_argument("--fs-method", default="PSO-GWO", help="features-json içindeki yöntem anahtarı")
args = ap.parse_args()
FS_TAG = f"_fs_{args.fs_method.lower().replace('-', '').replace(' ', '')}" if args.features_json else ""
TAG = ("_noage" if args.drop_age else "") + (f"_agemax{int(args.age_max)}" if args.age_max else "") + FS_TAG
OUT = C.out(f"02_models{TAG}")
RS = C.RANDOM_STATE

# ------------------------------------------------------------------ Veri
df = pd.read_csv(C.OUT_DIR / "00_data" / "clean_data.csv")
split = pd.read_csv(C.OUT_DIR / "00_data" / "split_indices.csv")
feats = [c for c in C.FEATURE_SETS[args.feature_set] if c in df.columns]
if args.features_json:
    feats = json.load(open(args.features_json))[args.fs_method]
    print(f"Öznitelikler {args.features_json} → {args.fs_method}")
if args.drop_age:
    feats = [c for c in feats if c != "Age"]
tr = split.loc[split.split == "train", "row_id"].values; te = split.loc[split.split == "test", "row_id"].values
if args.age_max is not None:
    tr = tr[df.loc[tr, "Age"].values <= args.age_max]; te = te[df.loc[te, "Age"].values <= args.age_max]
    print(f"Yaş filtresi Age<={args.age_max}: train={len(tr)}, test={len(te)}")
X_tr, y_tr = df.loc[tr, feats].values, df.loc[tr, C.TARGET].values
X_te, y_te = df.loc[te, feats].values, df.loc[te, C.TARGET].values
print(f"Öznitelik seti: {args.feature_set}{' (Age hariç)' if args.drop_age else ''} → {len(feats)} öznitelik: {feats}")
print(f"Train n={len(y_tr)} (TT={y_tr.sum()}), Test n={len(y_te)} (TT={y_te.sum()})  |  QUICK={C.QUICK}")
json.dump({"features": feats, "n_train": int(len(y_tr)), "n_test": int(len(y_te))}, open(OUT / "run_info.json", "w"), indent=2)

# ------------------------------------------------------------------ Örnekleyiciler
def make_sampler(name, k=5):
    k = int(k)
    return {
        "none": "passthrough",
        "ros": RandomOverSampler(random_state=RS),
        "smote": SMOTE(k_neighbors=k, random_state=RS),
        "borderline": BorderlineSMOTE(k_neighbors=k, random_state=RS),
        "svmsmote": SVMSMOTE(k_neighbors=k, random_state=RS),
        "adasyn": ADASYN(n_neighbors=k, random_state=RS),
        "smoteenn": SMOTEENN(smote=SMOTE(k_neighbors=k, random_state=RS), random_state=RS),
        "smotetomek": SMOTETomek(smote=SMOTE(k_neighbors=k, random_state=RS), random_state=RS),
    }[name]

# ------------------------------------------------------------------ Sınıflandırıcılar + arama uzayları
def make_clf(name, trial=None, params=None):
    """trial verilirse Optuna'dan örnekler; params verilirse sabit kurar. Döndürür: (estimator, params, needs_scaling)."""
    p = {} if params is None else dict(params)
    def S(key, kind, *a, **kw):
        if trial is not None:
            p[key] = getattr(trial, kind)(key, *a, **kw)
        return p[key]
    if name == "logreg":
        return LogisticRegression(C=S("C", "suggest_float", 1e-3, 1e2, log=True), penalty="elasticnet",
                                  l1_ratio=S("l1_ratio", "suggest_float", 0.0, 1.0), solver="saga", max_iter=5000, random_state=RS), p, True
    if name == "svm":
        return SVC(C=S("C", "suggest_float", 1e-2, 1e2, log=True), gamma=S("gamma", "suggest_float", 1e-4, 1.0, log=True),
                   kernel="rbf", probability=True, random_state=RS), p, True
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=S("n_neighbors", "suggest_int", 3, 41, step=2),
                                    weights=S("weights", "suggest_categorical", ["uniform", "distance"])), p, True
    if name in ("rf", "et"):
        Est = RandomForestClassifier if name == "rf" else ExtraTreesClassifier
        return Est(n_estimators=S("n_estimators", "suggest_int", 200, 800, step=100), max_depth=S("max_depth", "suggest_int", 2, 12),
                   min_samples_leaf=S("min_samples_leaf", "suggest_int", 1, 15), max_features=S("max_features", "suggest_float", 0.2, 1.0),
                   class_weight=S("class_weight", "suggest_categorical", [None, "balanced"]), n_jobs=1, random_state=RS), p, False
    if name == "gb":
        return GradientBoostingClassifier(n_estimators=S("n_estimators", "suggest_int", 50, 500, step=50),
                                          learning_rate=S("learning_rate", "suggest_float", 0.01, 0.3, log=True),
                                          max_depth=S("max_depth", "suggest_int", 1, 5), subsample=S("subsample", "suggest_float", 0.5, 1.0),
                                          min_samples_leaf=S("min_samples_leaf", "suggest_int", 1, 15), random_state=RS), p, False
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=S("n_estimators", "suggest_int", 50, 600, step=50),
                             learning_rate=S("learning_rate", "suggest_float", 0.01, 0.3, log=True), max_depth=S("max_depth", "suggest_int", 2, 8),
                             subsample=S("subsample", "suggest_float", 0.5, 1.0), colsample_bytree=S("colsample_bytree", "suggest_float", 0.4, 1.0),
                             min_child_weight=S("min_child_weight", "suggest_float", 1, 10), reg_lambda=S("reg_lambda", "suggest_float", 1e-2, 10, log=True),
                             gamma=S("gamma", "suggest_float", 0, 5), n_jobs=1, random_state=RS, eval_metric="logloss", verbosity=0, tree_method="hist"), p, False
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=S("n_estimators", "suggest_int", 50, 600, step=50),
                              learning_rate=S("learning_rate", "suggest_float", 0.01, 0.3, log=True), num_leaves=S("num_leaves", "suggest_int", 4, 48),
                              max_depth=S("max_depth", "suggest_int", 2, 8), min_child_samples=S("min_child_samples", "suggest_int", 5, 40),
                              subsample=S("subsample", "suggest_float", 0.5, 1.0), subsample_freq=1,
                              colsample_bytree=S("colsample_bytree", "suggest_float", 0.4, 1.0), reg_lambda=S("reg_lambda", "suggest_float", 1e-2, 10, log=True),
                              n_jobs=1, random_state=RS, verbose=-1), p, False
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=S("iterations", "suggest_int", 100, 800, step=100),
                                  learning_rate=S("learning_rate", "suggest_float", 0.01, 0.3, log=True), depth=S("depth", "suggest_int", 2, 8),
                                  l2_leaf_reg=S("l2_leaf_reg", "suggest_float", 1, 30, log=True), random_strength=S("random_strength", "suggest_float", 0, 5),
                                  bagging_temperature=S("bagging_temperature", "suggest_float", 0, 2),
                                  thread_count=1, random_seed=RS, verbose=0, allow_writing_files=False), p, False
    if name == "mlp":
        h = S("hidden", "suggest_categorical", ["16", "32", "64", "32-16", "64-32"])
        return MLPClassifier(hidden_layer_sizes=tuple(int(v) for v in str(h).split("-")), alpha=S("alpha", "suggest_float", 1e-5, 1e-1, log=True),
                             learning_rate_init=S("lr", "suggest_float", 1e-4, 1e-2, log=True), max_iter=1500, early_stopping=True, random_state=RS), p, True
    raise ValueError(name)

CLF_NAMES = {"logreg": "Logistic Regression (EN)", "svm": "SVM (RBF)", "knn": "k-NN", "rf": "Random Forest",
             "et": "Extra Trees", "gb": "Gradient Boosting", "xgb": "XGBoost", "lgbm": "LightGBM",
             "cat": "CatBoost", "mlp": "MLP"}

def build_pipeline(clf_name, params):
    clf, _, needs_scale = make_clf(clf_name, params=params)
    return ImbPipeline([
        ("scaler", StandardScaler() if needs_scale else "passthrough"),
        ("sampler", make_sampler(params["sampler"], params.get("k_neighbors", 5))),
        ("clf", clf),
    ])

def score_metric(y, prob):
    if C.OBJECTIVE_METRIC == "roc_auc": return roc_auc_score(y, prob)
    if C.OBJECTIVE_METRIC == "average_precision": return average_precision_score(y, prob)
    return matthews_corrcoef(y, (prob >= 0.5).astype(int))

def inner_objective(trial, clf_name, X, y):
    clf, params, needs_scale = make_clf(clf_name, trial=trial)
    params["sampler"] = trial.suggest_categorical("sampler", C.SAMPLERS)
    if params["sampler"] not in ("none", "ros"):
        params["k_neighbors"] = trial.suggest_int("k_neighbors", 3, 7)
    pipe = ImbPipeline([("scaler", StandardScaler() if needs_scale else "passthrough"),
                        ("sampler", make_sampler(params["sampler"], params.get("k_neighbors", 5))), ("clf", clf)])
    skf = StratifiedKFold(C.INNER_FOLDS, shuffle=True, random_state=RS + trial.number)
    scores = []
    for itr, iva in skf.split(X, y):
        try:
            m = clone(pipe).fit(X[itr], y[itr])
            scores.append(score_metric(y[iva], m.predict_proba(X[iva])[:, 1]))
        except Exception:
            return -1.0
    return float(np.mean(scores))

def tune(clf_name, X, y, seed):
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed, multivariate=True))
    study.optimize(lambda t: inner_objective(t, clf_name, X, y), n_trials=C.N_TRIALS, n_jobs=C.N_JOBS, show_progress_bar=False)
    bp = dict(study.best_params); bp.setdefault("k_neighbors", 5)
    return bp, study.best_value

# ------------------------------------------------------------------ Nested CV
outer = RepeatedStratifiedKFold(n_splits=C.OUTER_FOLDS, n_repeats=C.OUTER_REPEATS, random_state=RS)
fold_rows, oof_store = [], {}
t0 = time.time()
for clf_name in args.clf:
    oof = np.full((C.OUTER_REPEATS, len(y_tr)), np.nan)
    for fi, (otr, ova) in enumerate(outer.split(X_tr, y_tr)):
        rep = fi // C.OUTER_FOLDS
        bp, inner_best = tune(clf_name, X_tr[otr], y_tr[otr], seed=RS + fi)
        pipe = build_pipeline(clf_name, bp).fit(X_tr[otr], y_tr[otr])
        prob = pipe.predict_proba(X_tr[ova])[:, 1]
        oof[rep, ova] = prob
        m = binary_metrics(y_tr[ova], prob, 0.5)
        m.update({"clf": clf_name, "outer_fold": fi, "repeat": rep, "inner_best": inner_best,
                  "sampler": bp["sampler"], "params": json.dumps(bp)})
        fold_rows.append(m)
        print(f"[{clf_name:6s}] dış kat {fi+1:2d}/{C.OUTER_FOLDS*C.OUTER_REPEATS}  AUC={m['AUC']:.3f} MCC={m['MCC']:.3f} "
              f"sampler={bp['sampler']:10s}  ({(time.time()-t0)/60:.1f} dk)")
    oof_store[clf_name] = oof
    pd.DataFrame(fold_rows).to_csv(OUT / "nested_cv_fold_results.csv", index=False)   # ara kayıt

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT / "nested_cv_fold_results.csv", index=False)

# Özet tablo: mean ± SD ve %95 GA (t-dağılımı, n = dış kat sayısı)
from scipy import stats as st
metrics_cols = ["AUC", "AUPRC", "Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1", "MCC", "Brier"]
summ = []
for clf_name, g in fold_df.groupby("clf"):
    row = {"Model": CLF_NAMES[clf_name], "clf": clf_name, "n_outer": len(g)}
    for mc in metrics_cols:
        v = g[mc].astype(float).dropna(); mu, sd = v.mean(), v.std(ddof=1)
        h = st.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else 0
        row[mc] = mu; row[f"{mc}_SD"] = sd; row[f"{mc}_CI"] = f"{mu:.3f} [{mu-h:.3f}, {mu+h:.3f}]"
    row["most_frequent_sampler"] = g["sampler"].mode().iloc[0]
    row["sampler_counts"] = json.dumps(g["sampler"].value_counts().to_dict())
    summ.append(row)
summ = pd.DataFrame(summ).sort_values("MCC", ascending=False)
summ.to_excel(OUT / "Table4_nested_cv_summary.xlsx", index=False)
print("\nNESTED CV ÖZETİ:")
print(summ[["Model", "AUC_CI", "MCC_CI", "Sensitivity_CI", "Specificity_CI", "most_frequent_sampler"]].to_string(index=False))

# Nested CV boxplot
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
order = summ["clf"].tolist()
for ax, mc in zip(axes, ["AUC", "MCC", "Balanced_Accuracy"]):
    sns.boxplot(data=fold_df, x="clf", y=mc, order=order, ax=ax, color="#9ecae1", width=0.6, fliersize=2)
    sns.stripplot(data=fold_df, x="clf", y=mc, order=order, ax=ax, color="k", size=2.5, alpha=.6)
    ax.set_xticklabels([CLF_NAMES[c] for c in order], rotation=45, ha="right", fontsize=8); ax.set_xlabel("")
    ax.set_title(f"Nested CV — {mc.replace('_',' ')}")
plt.tight_layout(); plt.savefig(OUT / "Fig_nested_cv_boxplot.png", dpi=300); plt.close()

# ------------------------------------------------------------------ Nihai modeller + hold-out test
best_cfg, test_rows, preds = {}, [], {"row_id": te, "y_true": y_te}
oof_final = {}
for clf_name in args.clf:
    bp, _ = tune(clf_name, X_tr, y_tr, seed=RS + 999)
    best_cfg[clf_name] = bp
    pipe = build_pipeline(clf_name, bp).fit(X_tr, y_tr)
    joblib.dump(pipe, C.out(f"02_models{TAG}/final_models") / f"{clf_name}.joblib")
    # Youden eşiği: eğitim setinde OOF (dış tekrar ortalaması) tahminlerinden — test bilgisi kullanılmaz
    oof_mean = np.nanmean(oof_store[clf_name], axis=0)
    oof_final[clf_name] = oof_mean
    thr_y, _, _ = youden_threshold(y_tr, oof_mean)
    prob = pipe.predict_proba(X_te)[:, 1]
    preds[f"prob_{clf_name}"] = prob
    for thr_name, thr in (("0.5", 0.5), ("Youden(train-OOF)", thr_y)):
        bm = bootstrap_metrics(y_te, prob, thr, n_boot=C.N_BOOT, seed=RS)
        auc, lo, hi, _ = delong_ci(y_te, prob)
        row = {"Model": CLF_NAMES[clf_name], "clf": clf_name, "threshold_rule": thr_name, "threshold": thr,
               "sampler": bp["sampler"], "AUC_DeLong_CI": fmt_ci(auc, lo, hi)}
        for mc in metrics_cols + ["TP", "TN", "FP", "FN"]:
            row[mc] = bm.loc[mc, "Estimate"]
            if mc in metrics_cols:
                row[f"{mc}_CI"] = fmt_ci(bm.loc[mc, "Estimate"], bm.loc[mc, "CI_low"], bm.loc[mc, "CI_high"])
        test_rows.append(row)
    print(f"[{clf_name:6s}] HOLD-OUT  AUC={roc_auc_score(y_te, prob):.3f}  MCC@0.5={matthews_corrcoef(y_te, prob>=0.5):.3f}  sampler={bp['sampler']}")

json.dump(best_cfg, open(OUT / "best_configs.json", "w"), indent=2)
pd.DataFrame(preds).to_csv(OUT / "test_predictions.csv", index=False)
pd.DataFrame({"row_id": tr, "y_true": y_tr, **{f"oof_{k}": v for k, v in oof_final.items()}}).to_csv(OUT / "oof_predictions.csv", index=False)
test_df = pd.DataFrame(test_rows)
test_df.to_excel(OUT / "Table5_holdout_test.xlsx", index=False)
print("\nHOLD-OUT TEST (eşik 0.5):")
print(test_df[test_df.threshold_rule == "0.5"][["Model", "AUC_CI", "MCC_CI", "Sensitivity_CI", "Specificity_CI", "PPV_CI", "NPV_CI"]].to_string(index=False))

# Kıyas: yalnız-Age lojistik regresyon (hematolojik belirteçlerin artımlı değeri için referans)
if "Age" in feats:
    ai = feats.index("Age")
    from sklearn.model_selection import cross_val_predict
    lr_age = LogisticRegression(max_iter=1000).fit(X_tr[:, [ai]], y_tr)
    p_age = lr_age.predict_proba(X_te[:, [ai]])[:, 1]
    preds["prob_age_only"] = p_age
    oof_age = cross_val_predict(LogisticRegression(max_iter=1000), X_tr[:, [ai]], y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=RS), method="predict_proba")[:, 1]
    thr_age, _, _ = youden_threshold(y_tr, oof_age)
    for thr_name, thr in (("0.5", 0.5), ("Youden(train-OOF)", thr_age)):
        bm = bootstrap_metrics(y_te, p_age, thr, n_boot=C.N_BOOT, seed=RS); auc, lo, hi, _ = delong_ci(y_te, p_age)
        row = {"Model": "Age-only logistic (reference)", "clf": "age_only", "threshold_rule": thr_name, "threshold": thr, "sampler": "none", "AUC_DeLong_CI": fmt_ci(auc, lo, hi)}
        for mc in metrics_cols + ["TP", "TN", "FP", "FN"]:
            row[mc] = bm.loc[mc, "Estimate"]
            if mc in metrics_cols: row[f"{mc}_CI"] = fmt_ci(bm.loc[mc, "Estimate"], bm.loc[mc, "CI_low"], bm.loc[mc, "CI_high"])
        test_rows.append(row)
    test_df = pd.DataFrame(test_rows); test_df.to_excel(OUT / "Table5_holdout_test.xlsx", index=False)
    pd.DataFrame(preds).to_csv(OUT / "test_predictions.csv", index=False)
    print(f"[age_only] HOLD-OUT  AUC={roc_auc_score(y_te, p_age):.3f}  (referans)")

# En iyi model: nested CV MCC'ye göre (test setine bakılmadan seçim!)
best = summ.iloc[0]["clf"]
json.dump({"best_model": best, "criterion": f"highest nested-CV {C.OBJECTIVE_METRIC.upper()} mean (selected without test set)"},
          open(OUT / "best_model.json", "w"), indent=2)
print(f"\nEN İYİ MODEL (nested CV'ye göre): {CLF_NAMES[best]}")

# DeLong: en iyi model vs diğerleri (test setinde)
dl = []
for clf_name in args.clf + (["age_only"] if "prob_age_only" in preds else []):
    if clf_name == best: continue
    z, p, a1, a2 = delong_test(y_te, preds[f"prob_{best}"], preds[f"prob_{clf_name}"])
    dl.append({"Reference": CLF_NAMES[best], "Comparator": CLF_NAMES.get(clf_name, "Age-only logistic"), "AUC_ref": a1, "AUC_comp": a2, "z": z, "p": p})
pd.DataFrame(dl).to_excel(OUT / "Table6_delong_vs_best.xlsx", index=False)

# ------------------------------------------------------------------ Tablo 7: örnekleyici karşılaştırması (en iyi model, sabit HP)
samp_rows = []
bp_best = dict(best_cfg[best])
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1 if C.QUICK else 5, random_state=RS)
for s in C.SAMPLERS:
    cfg = dict(bp_best); cfg["sampler"] = s
    mets = []
    for itr, iva in rskf.split(X_tr, y_tr):
        try:
            m = build_pipeline(best, cfg).fit(X_tr[itr], y_tr[itr])
            mets.append(binary_metrics(y_tr[iva], m.predict_proba(X_tr[iva])[:, 1]))
        except Exception as e:
            print(f"  sampler {s} hata: {e}")
    if mets:
        md = pd.DataFrame(mets)
        row = {"Sampler": s}
        for mc in ["AUC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "F1", "MCC", "Brier"]:
            row[mc] = md[mc].mean(); row[f"{mc}_SD"] = md[mc].std(ddof=1)
        # hold-out test (bilgi amaçlı — seçim CV'ye göre yapılır)
        mt = build_pipeline(best, cfg).fit(X_tr, y_tr)
        tm = binary_metrics(y_te, mt.predict_proba(X_te)[:, 1])
        row.update({"Test_AUC": tm["AUC"], "Test_MCC": tm["MCC"], "Test_Sens": tm["Sensitivity"], "Test_Spec": tm["Specificity"]})
        samp_rows.append(row)
samp_df = pd.DataFrame(samp_rows).sort_values("MCC", ascending=False)
samp_df.to_excel(OUT / "Table7_sampler_comparison.xlsx", index=False)
print("\nÖRNEKLEYİCİ KARŞILAŞTIRMASI (en iyi model, CV):")
print(samp_df[["Sampler", "AUC", "MCC", "Sensitivity", "Specificity", "Test_AUC", "Test_MCC"]].round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 3.8))
sd = samp_df.melt(id_vars="Sampler", value_vars=["AUC", "MCC", "Sensitivity", "Specificity"])
sns.barplot(data=sd, x="Sampler", y="value", hue="variable", ax=ax); ax.set_ylim(0, 1); ax.set_ylabel("Score (CV mean)")
ax.set_title(f"Resampling strategies — {CLF_NAMES[best]} (training-set CV)"); ax.legend(fontsize=8, ncol=4, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "Fig_sampler_comparison.png", dpi=300); plt.close()

# ------------------------------------------------------------------ Şekiller: ROC, PR, kalibrasyon, DCA, karışıklık matrisleri
fig, ax = plt.subplots(figsize=(5.6, 5.3))
for clf_name in args.clf:
    p_ = preds[f"prob_{clf_name}"]; fpr, tpr, _ = roc_curve(y_te, p_)
    ax.plot(fpr, tpr, lw=2.2 if clf_name == best else 1.2, alpha=1 if clf_name == best else .75,
            label=f"{CLF_NAMES[clf_name]} (AUC {roc_auc_score(y_te, p_):.3f})")
if "prob_age_only" in preds:
    fpr, tpr, _ = roc_curve(y_te, preds["prob_age_only"]); ax.plot(fpr, tpr, color="grey", ls="-.", lw=1.4, label=f"Age-only logistic (AUC {roc_auc_score(y_te, preds['prob_age_only']):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title(f"ROC — hold-out test (n={len(y_te)})"); ax.legend(fontsize=7.5, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "Fig_ROC_test.png", dpi=300); plt.savefig(OUT / "Fig_ROC_test.pdf"); plt.close()

fig, ax = plt.subplots(figsize=(5.6, 5.3))
for clf_name in args.clf:
    p_ = preds[f"prob_{clf_name}"]; pr, rc, _ = precision_recall_curve(y_te, p_)
    ax.plot(rc, pr, lw=2.2 if clf_name == best else 1.2, label=f"{CLF_NAMES[clf_name]} (AP {average_precision_score(y_te, p_):.3f})")
ax.axhline(y_te.mean(), color="k", ls="--", lw=1, label=f"Prevalence ({y_te.mean():.2f})")
ax.set_xlabel("Recall (Sensitivity)"); ax.set_ylabel("Precision (PPV)"); ax.set_title("Precision–Recall — hold-out test"); ax.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig(OUT / "Fig_PR_test.png", dpi=300); plt.close()

fig, ax = plt.subplots(figsize=(5.2, 5))
for clf_name in args.clf:
    p_ = preds[f"prob_{clf_name}"]
    fp_, mp_ = calibration_curve(y_te, p_, n_bins=8, strategy="quantile")
    ax.plot(mp_, fp_, "o-", lw=2 if clf_name == best else 1, ms=4, label=f"{CLF_NAMES[clf_name]} (Brier {np.mean((p_-y_te)**2):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("Calibration — hold-out test"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "Fig_calibration.png", dpi=300); plt.close()

fig, ax = plt.subplots(figsize=(6, 4.2))
dc = decision_curve(y_te, preds[f"prob_{best}"])
ax.plot(dc.threshold, dc.NB_model, lw=2, label=CLF_NAMES[best])
for clf_name in args.clf:
    if clf_name != best:
        ax.plot(dc.threshold, decision_curve(y_te, preds[f"prob_{clf_name}"]).NB_model, lw=.9, alpha=.6, label=CLF_NAMES[clf_name])
ax.plot(dc.threshold, dc.NB_treat_all, "k--", lw=1, label="Treat all"); ax.plot(dc.threshold, dc.NB_treat_none, "k:", lw=1, label="Treat none")
ax.set_ylim(-0.05, max(0.3, dc.NB_model.max() + 0.05)); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
ax.set_title("Decision curve analysis — hold-out test"); ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig(OUT / "Fig_DCA.png", dpi=300); plt.close()
dc.to_csv(OUT / "decision_curve_best.csv", index=False)

n = len(args.clf); ncol = 5; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.8 * nrow))
for ax, clf_name in zip(np.ravel(axes), args.clf):
    p_ = preds[f"prob_{clf_name}"]; m = binary_metrics(y_te, p_)
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax, xticklabels=["EO", "TT"], yticklabels=["EO", "TT"])
    ax.set_title(f"{CLF_NAMES[clf_name]}\nMCC {m['MCC']:.2f}", fontsize=9); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
for ax in np.ravel(axes)[n:]: ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "Fig_confusion_matrices.png", dpi=300); plt.close()

print(f"\nToplam süre: {(time.time()-t0)/60:.1f} dk  →  {OUT}")
