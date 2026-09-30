# -*- coding: utf-8 -*-
"""
ADIM 3 — Açıklanabilir yapay zekâ (XAI) ve çok-yöntemli öznitelik önemi.

En iyi model (02 adımında nested-CV MCC'ye göre seçilen; --model ile değiştirilebilir) için:
  (a) SHAP  — TreeExplainer (ağaç modelleri) veya Permutation/Kernel explainer (diğerleri); test setinde
  (b) LIME  — her test hastası için yerel açıklama, |ağırlık| ortalaması ile küresel sıralama
  (c) Permütasyon önemi — test setinde, AUC kaybı (n_repeats)
  (d) Model-içi önem — feature_importances_ / |coef| (varsa)
  (e) PDP + ICE (1-B, en önemli 6) ve 2-B PDP (en güçlü 3 çift)
  (f) SHAP etkileşim değerleri (ağaç modelleri)
  (g) SHAP-tabanlı klinik eşik: SHAP=0 geçiş noktası (risk yönü değiştiği değer)
  (h) Yöntemler arası sıralama uyumu: ortalama sıra, sıra SD (σ<1 yüksek, 1–2 orta, ≥2 düşük), Kendall W

Kullanım:
  python 03_xai.py                    # 02_models/ içindeki en iyi model
  python 03_xai.py --model cat        # belirli model
  python 03_xai.py --drop-age         # 02_models_noage/ çıktıları üzerinde

Çıktılar (OUT_DIR/03_xai[_noage]/): Table8_feature_importance_all_methods.xlsx, Table9_shap_thresholds.xlsx,
  Fig_SHAP_beeswarm/bar/dependence/waterfall/interaction, Fig_LIME_global, Fig_permutation, Fig_PDP_ICE, Fig_PDP_2D,
  Fig_importance_rank_agreement, shap_values_test.csv, lime_weights_test.csv
"""
import argparse, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance, PartialDependenceDisplay, partial_dependence
from sklearn.metrics import roc_auc_score
from scipy import stats
import config as C
from utils import binary_metrics

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None)
ap.add_argument("--drop-age", action="store_true")
ap.add_argument("--age-max", type=float, default=None, help="yalnız Age<=N hastalar (yaş-örtüşen alt kohort duyarlılık analizi)")
ap.add_argument("--tag", default=None, help="02 çıktı klasörü son eki, örn. _fs_psogwo (verilirse --drop-age/--age-max yerine geçer)")
ap.add_argument("--n-repeats", type=int, default=50)
ap.add_argument("--lime-samples", type=int, default=1000)
args = ap.parse_args()
TAG = args.tag if args.tag is not None else ("_noage" if args.drop_age else "") + (f"_agemax{int(args.age_max)}" if args.age_max else "")
MODEL_DIR = C.OUT_DIR / f"02_models{TAG}"
OUT = C.out(f"03_xai{TAG}")
RS = C.RANDOM_STATE
if C.QUICK:
    args.n_repeats, args.lime_samples = 5, 200

# ------------------------------------------------------------------ Veri ve model
info = json.load(open(MODEL_DIR / "run_info.json")); feats = info["features"]
pretty = [C.PRETTY.get(f, f).split(" (")[0] for f in feats]
df = pd.read_csv(C.OUT_DIR / "00_data" / "clean_data.csv")
split = pd.read_csv(C.OUT_DIR / "00_data" / "split_indices.csv")
tr = split.loc[split.split == "train", "row_id"].values; te = split.loc[split.split == "test", "row_id"].values
if args.age_max is not None:
    tr = tr[df.loc[tr, "Age"].values <= args.age_max]; te = te[df.loc[te, "Age"].values <= args.age_max]
X_tr = df.loc[tr, feats].astype(float).reset_index(drop=True); y_tr = df.loc[tr, C.TARGET].values
X_te = df.loc[te, feats].astype(float).reset_index(drop=True); y_te = df.loc[te, C.TARGET].values
best = args.model or json.load(open(MODEL_DIR / "best_model.json"))["best_model"]
pipe = joblib.load(MODEL_DIR / "final_models" / f"{best}.joblib")
clf = pipe.named_steps["clf"]; scaler = pipe.named_steps["scaler"]
TREE = type(clf).__name__ in ("RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier",
                              "XGBClassifier", "LGBMClassifier", "CatBoostClassifier")
print(f"Model: {best} ({type(clf).__name__}), tree={TREE}, öznitelik={len(feats)}")
prob_te = pipe.predict_proba(X_te.values)[:, 1]
print(f"Test AUC = {roc_auc_score(y_te, prob_te):.3f}")

def transform(X):  # ölçekleyici varsa uygula (ağaç modellerinde passthrough)
    return X.values if scaler == "passthrough" or scaler is None else scaler.transform(X.values)

# ------------------------------------------------------------------ (a) SHAP
if TREE:
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(transform(X_te))
    if isinstance(sv, list): sv = sv[1]
    elif np.ndim(sv) == 3: sv = sv[:, :, 1]
    base = explainer.expected_value
    base = base[1] if np.ndim(base) else base
    unit = "log-odds" if type(clf).__name__ in ("XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "GradientBoostingClassifier") else "probability"
else:
    bg = shap.maskers.Independent(X_tr.values, max_samples=100)
    f = lambda X: pipe.predict_proba(X)[:, 1]
    explainer = shap.Explainer(f, bg, algorithm="permutation", seed=RS)
    exp = explainer(X_te.values, max_evals=2 * len(feats) + 1)
    sv, base, unit = exp.values, float(np.mean(exp.base_values)), "probability"
shap_exp = shap.Explanation(values=sv, base_values=np.full(len(sv), base), data=X_te.values, feature_names=pretty)
pd.DataFrame(sv, columns=feats).assign(row_id=te, y_true=y_te, prob=prob_te).to_csv(OUT / "shap_values_test.csv", index=False)
shap_imp = pd.Series(np.abs(sv).mean(0), index=feats)
shap_pct = 100 * shap_imp / shap_imp.sum()

plt.figure(figsize=(7, 5.5)); shap.plots.beeswarm(shap_exp, max_display=len(feats), show=False)
plt.title(f"SHAP summary — {best} (test set, {unit})"); plt.tight_layout(); plt.savefig(OUT / "Fig_SHAP_beeswarm.png", dpi=300, bbox_inches="tight"); plt.close()
plt.figure(figsize=(6, 5)); shap.plots.bar(shap_exp, max_display=len(feats), show=False)
plt.title("Mean |SHAP| — global importance"); plt.tight_layout(); plt.savefig(OUT / "Fig_SHAP_bar.png", dpi=300, bbox_inches="tight"); plt.close()

top = shap_imp.sort_values(ascending=False).index.tolist()
# Bağımlılık grafikleri (en iyi 6)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, f in zip(axes.ravel(), top[:6]):
    i = feats.index(f)
    inter = feats.index(top[0]) if f != top[0] else feats.index(top[1])
    sc = ax.scatter(X_te[f], sv[:, i], c=X_te.iloc[:, inter], cmap="coolwarm", s=14, alpha=.8)
    ax.axhline(0, color="k", lw=.8, ls="--"); ax.set_xlabel(C.PRETTY.get(f, f)); ax.set_ylabel(f"SHAP ({unit})")
    if X_te[f].skew() > 2: ax.set_xscale("log")
    cb = plt.colorbar(sc, ax=ax); cb.set_label(feats[inter], fontsize=8)
plt.suptitle("SHAP dependence plots (colour = strongest other feature)"); plt.tight_layout()
plt.savefig(OUT / "Fig_SHAP_dependence.png", dpi=300); plt.close()

# Waterfall: TP / TN / FP / FN örnekleri
pred = (prob_te >= 0.5).astype(int)
cases = {"True positive (TT correctly predicted)": (y_te == 1) & (pred == 1), "True negative (EO correctly predicted)": (y_te == 0) & (pred == 0),
         "False positive (EO predicted as TT)": (y_te == 0) & (pred == 1), "False negative (TT predicted as EO)": (y_te == 1) & (pred == 0)}
for k, (title, mask) in enumerate(cases.items()):
    idx = np.where(mask)[0]
    if len(idx) == 0: continue
    i = idx[np.argmax(np.abs(prob_te[idx] - 0.5))] if "correct" in title else idx[0]
    plt.figure(figsize=(7, 5)); shap.plots.waterfall(shap_exp[i], max_display=12, show=False)
    plt.title(f"{title}\nrow_id={te[i]}, p(TT)={prob_te[i]:.2f}", fontsize=10); plt.tight_layout()
    plt.savefig(OUT / f"Fig_SHAP_waterfall_{k+1}_{title.split(' (')[0].replace(' ', '_')}.png", dpi=300, bbox_inches="tight"); plt.close()

# (f) SHAP etkileşimleri
if TREE and type(clf).__name__ != "GradientBoostingClassifier":
    try:
        iv = explainer.shap_interaction_values(transform(X_te))
        if isinstance(iv, list): iv = iv[1]
        elif np.ndim(iv) == 4: iv = iv[:, :, :, 1]
        mi = np.abs(iv).mean(0); np.fill_diagonal(mi, 0)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(pd.DataFrame(mi, index=pretty, columns=pretty), cmap="viridis", ax=ax, square=True, cbar_kws={"shrink": .7})
        ax.set_title("Mean |SHAP interaction| (off-diagonal)"); plt.tight_layout(); plt.savefig(OUT / "Fig_SHAP_interaction.png", dpi=300); plt.close()
        pairs = [(feats[i], feats[j], mi[i, j]) for i in range(len(feats)) for j in range(i + 1, len(feats))]
        pd.DataFrame(pairs, columns=["A", "B", "mean_abs_interaction"]).sort_values("mean_abs_interaction", ascending=False).to_csv(OUT / "shap_interactions.csv", index=False)
    except Exception as e:
        print("SHAP etkileşim atlandı:", e)

# ------------------------------------------------------------------ (g) SHAP tabanlı eşik (SHAP=0 geçişi)
thr_rows = []
for f in top[:8]:
    i = feats.index(f); x = X_te[f].values; s = sv[:, i]
    o = np.argsort(x); xs, ss = x[o], s[o]
    # hareketli ortalama (pencere ~15) ile sıfır geçişi
    w = max(5, len(xs) // 8); sm = pd.Series(ss).rolling(w, center=True, min_periods=3).mean().values
    cross = np.where(np.diff(np.sign(sm[~np.isnan(sm)])) != 0)[0]
    xv = xs[~np.isnan(sm)]
    thr = float(np.mean([xv[c] for c in cross])) if len(cross) else np.nan
    rho, p = stats.spearmanr(x, s)
    thr_rows.append({"Feature": C.PRETTY.get(f, f), "SHAP_zero_crossing": thr, "n_crossings": len(cross),
                     "direction": "higher → TT" if rho > 0 else "lower → TT", "Spearman(x, SHAP)": rho, "p": p,
                     "mean|SHAP|": shap_imp[f], "SHAP_%": shap_pct[f]})
pd.DataFrame(thr_rows).to_excel(OUT / "Table9_shap_thresholds.xlsx", index=False)

# ------------------------------------------------------------------ (b) LIME
lime_exp = LimeTabularExplainer(X_tr.values, feature_names=feats, class_names=["EO", "TT"], mode="classification",
                                discretize_continuous=True, random_state=RS)
W = np.zeros((len(X_te), len(feats)))
for r in range(len(X_te)):
    e = lime_exp.explain_instance(X_te.values[r], lambda X: pipe.predict_proba(X), num_features=len(feats), num_samples=args.lime_samples)
    for j, w in e.as_map()[1]:
        W[r, j] = w
pd.DataFrame(W, columns=feats).assign(row_id=te).to_csv(OUT / "lime_weights_test.csv", index=False)
lime_imp = pd.Series(np.abs(W).mean(0), index=feats)
fig, ax = plt.subplots(figsize=(6, 5)); o = lime_imp.sort_values()
ax.barh([C.PRETTY.get(f, f).split(" (")[0] for f in o.index], o.values, color="#8da0cb"); ax.set_xlabel("Mean |LIME weight| (test set)")
ax.set_title("LIME global importance"); plt.tight_layout(); plt.savefig(OUT / "Fig_LIME_global.png", dpi=300); plt.close()

# ------------------------------------------------------------------ (c) Permütasyon önemi
pi = permutation_importance(pipe, X_te.values, y_te, scoring="roc_auc", n_repeats=args.n_repeats, random_state=RS, n_jobs=C.N_JOBS)
perm_imp = pd.Series(pi.importances_mean, index=feats); perm_sd = pd.Series(pi.importances_std, index=feats)
fig, ax = plt.subplots(figsize=(6, 5)); o = perm_imp.sort_values()
ax.barh([C.PRETTY.get(f, f).split(" (")[0] for f in o.index], o.values, xerr=perm_sd[o.index].values, color="#66c2a5", error_kw={"lw": .8})
ax.set_xlabel(f"Mean AUC decrease ({args.n_repeats} permutations, test set)"); ax.set_title("Permutation importance")
plt.tight_layout(); plt.savefig(OUT / "Fig_permutation.png", dpi=300); plt.close()

# ------------------------------------------------------------------ (d) Model-içi önem
model_imp = pd.Series(np.nan, index=feats)
if hasattr(clf, "feature_importances_"):
    model_imp = pd.Series(clf.feature_importances_, index=feats)
elif hasattr(clf, "coef_"):
    model_imp = pd.Series(np.abs(np.ravel(clf.coef_)), index=feats)
if model_imp.notna().all() and model_imp.sum() > 0:
    model_imp = model_imp / model_imp.sum()

# ------------------------------------------------------------------ (h) Sıralama uyumu
imp = pd.DataFrame({"SHAP": shap_imp, "LIME": lime_imp, "Permutation": perm_imp, "Model-based": model_imp})
methods = [m for m in imp.columns if imp[m].notna().all()]
ranks = imp[methods].rank(ascending=False)
imp["SHAP_%"] = shap_pct
imp["mean_rank"] = ranks.mean(1); imp["rank_SD"] = ranks.std(1, ddof=0)
imp["agreement"] = imp["rank_SD"].map(lambda s: "high (σ<1)" if s < 1 else "moderate (1≤σ<2)" if s < 2 else "low (σ≥2)")
for m in methods: imp[f"rank_{m}"] = ranks[m]
imp.index = [C.PRETTY.get(f, f) for f in imp.index]
imp = imp.sort_values("mean_rank")
k = len(feats); n_m = len(methods)
Wk = 12 * ((ranks.sum(1) - n_m * (k + 1) / 2) ** 2).sum() / (n_m ** 2 * (k ** 3 - k))
chi2 = n_m * (k - 1) * Wk; p_w = stats.chi2.sf(chi2, k - 1)
imp.attrs["kendall_W"] = Wk
with pd.ExcelWriter(OUT / "Table8_feature_importance_all_methods.xlsx") as xw:
    imp.to_excel(xw, sheet_name="importance")
    pd.DataFrame({"Kendall_W": [Wk], "chi2": [chi2], "p": [p_w], "methods": [", ".join(methods)], "model": [best]}).to_excel(xw, sheet_name="agreement", index=False)
print(f"\nÖZNİTELİK ÖNEMİ (yöntemler: {methods}); Kendall W = {Wk:.3f}, p = {p_w:.4f}")
print(imp[["SHAP_%", "mean_rank", "rank_SD", "agreement"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 5))
rk = ranks.copy(); rk.index = [C.PRETTY.get(f, f).split(" (")[0] for f in rk.index]
rk = rk.loc[[C.PRETTY.get(f, f).split(" (")[0] for f in shap_imp.sort_values(ascending=False).index]]
sns.heatmap(rk, annot=True, fmt=".0f", cmap="YlGnBu_r", cbar_kws={"label": "rank (1 = most important)"}, ax=ax)
ax.set_title(f"Feature rank across XAI methods (Kendall W = {Wk:.2f})"); plt.tight_layout()
plt.savefig(OUT / "Fig_importance_rank_agreement.png", dpi=300); plt.close()

# ------------------------------------------------------------------ (e) PDP + ICE
top6 = top[:6]
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
PartialDependenceDisplay.from_estimator(pipe, X_tr, features=[feats.index(f) for f in top6], feature_names=feats, kind="both",
                                        centered=True, subsample=100, ice_lines_kw={"alpha": .12, "lw": .6}, pd_line_kw={"color": "crimson", "lw": 2.2},
                                        response_method="predict_proba", ax=axes.ravel(), random_state=RS)
for ax, f in zip(axes.ravel(), top6):
    ax.set_xlabel(C.PRETTY.get(f, f)); ax.set_ylabel("Centred partial dependence\nP(TT)")
plt.suptitle("Partial dependence (red) with ICE curves (grey) — training set"); plt.tight_layout()
plt.savefig(OUT / "Fig_PDP_ICE.png", dpi=300); plt.close()

# ICE heterojenliği: her x-noktasında ICE eğrilerinin SD'sinin ortalaması
ice_rows = []
for f in top6:
    r = partial_dependence(pipe, X_tr, [feats.index(f)], kind="individual", response_method="predict_proba", grid_resolution=30)
    ind = r["individual"][0]                       # (n_samples, n_grid)
    ind = ind - ind[:, [0]]                         # merkezle
    ice_rows.append({"Feature": C.PRETTY.get(f, f), "PDP_range": float(np.ptp(ind.mean(0))),
                     "ICE_heterogeneity(mean SD)": float(ind.std(0).mean()), "ICE_heterogeneity(max SD)": float(ind.std(0).max())})
pd.DataFrame(ice_rows).to_excel(OUT / "Table10_PDP_ICE_summary.xlsx", index=False)

# 2-B PDP (en güçlü 3 çift; etkileşim dosyası varsa oradan, yoksa SHAP top-3 kombinasyonları)
pairs = [(top[0], top[1]), (top[0], top[2]), (top[1], top[2])]
if (OUT / "shap_interactions.csv").exists():
    si = pd.read_csv(OUT / "shap_interactions.csv").head(3)
    pairs = list(zip(si.A, si.B))
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
PartialDependenceDisplay.from_estimator(pipe, X_tr, features=[(feats.index(a), feats.index(b)) for a, b in pairs], feature_names=feats,
                                        response_method="predict_proba", ax=axes, grid_resolution=25, contour_kw={"cmap": "RdYlBu_r"})
plt.suptitle("Two-way partial dependence — P(TT)"); plt.tight_layout(); plt.savefig(OUT / "Fig_PDP_2D.png", dpi=300); plt.close()

# ------------------------------------------------------------------ Özet json
json.dump({"model": best, "shap_unit": unit, "shap_pct": shap_pct.round(2).to_dict(), "kendall_W": Wk, "kendall_p": p_w,
           "top_features": top[:6], "test_metrics": {k: float(v) for k, v in binary_metrics(y_te, prob_te).items()}},
          open(OUT / "xai_summary.json", "w"), indent=2)
print(f"\nBitti → {OUT}")
