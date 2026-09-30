# -*- coding: utf-8 -*-
"""
ADIM 5 — Lojistik regresyon nomogramı + kalibrasyon + basit puan skoru.

Öznitelikler: 04 adımının PSO-GWO seçimi (varsayılan) veya --features ile elle.
Model eğitim setinde kurulur; ayrım/kalibrasyon hold-out testte raporlanır.

Çıktılar (OUT_DIR/05_nomogram[tag]/):
  Table13_nomogram_logistic.xlsx : β, SE, OR (95% CI), p, VIF; nomogram puan tablosu (0–100 ölçeği)
  Table14_nomogram_performance.xlsx : train/test AUC (DeLong CI), Brier, Hosmer–Lemeshow, kalibrasyon eğim/kesişim
  Table15_simple_score.xlsx : Youden eşiklerine dayalı 0–k puan; puana göre gözlenen TT oranı (train ve test)
  Fig_nomogram.png, Fig_nomogram_calibration.png, Fig_score_TT_rate.png
  nomogram_model.json : web uygulaması için katsayılar ve ölçekler
"""
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import config as C
from utils import delong_ci, binary_metrics, youden_threshold, bootstrap_metrics

sns.set_theme(style="whitegrid", context="paper")
ap = argparse.ArgumentParser()
ap.add_argument("--features", nargs="*", default=None)
ap.add_argument("--features-json", default=None)
ap.add_argument("--fs-method", default="PSO-GWO")
ap.add_argument("--tag", default="")
args = ap.parse_args()
OUT = C.out(f"05_nomogram{args.tag}")
RS = C.RANDOM_STATE

df = pd.read_csv(C.OUT_DIR / "00_data" / "clean_data.csv")
split = pd.read_csv(C.OUT_DIR / "00_data" / "split_indices.csv")
if args.features: feats = args.features
else:
    fj = args.features_json or (C.OUT_DIR / "04_feature_selection" / "selected_features.json")
    feats = json.load(open(fj))[args.fs_method]
tr = split.loc[split.split == "train", "row_id"].values; te = split.loc[split.split == "test", "row_id"].values
Xtr, ytr = df.loc[tr, feats].astype(float), df.loc[tr, C.TARGET].values
Xte, yte = df.loc[te, feats].astype(float), df.loc[te, C.TARGET].values
print(f"Nomogram öznitelikleri ({len(feats)}): {feats}")

# ------------------------------------------------------------------ Lojistik model (statsmodels: SE/p; sklearn: aynı katsayılar, ridge yok)
Xc = sm.add_constant(Xtr)
res = sm.Logit(ytr, Xc).fit(disp=0, maxiter=500)
coef = pd.DataFrame({"Variable": ["Intercept"] + [C.PRETTY.get(f, f) for f in feats], "code": ["const"] + feats,
                     "beta": res.params.values, "SE": res.bse.values, "z": res.tvalues.values, "p": res.pvalues.values,
                     "OR": np.exp(res.params.values), "OR_CI_low": np.exp(res.conf_int()[0].values), "OR_CI_high": np.exp(res.conf_int()[1].values)})
coef.loc[1:, "VIF"] = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
# Standardize edilmiş β (per 1 SD) — önem sıralaması için
sd = Xtr.std(ddof=1).values
coef.loc[1:, "beta_std(per SD)"] = res.params.values[1:] * sd
print(coef.round(4).to_string(index=False))

# ------------------------------------------------------------------ Nomogram puanları (0–100 ölçek, en güçlü değişken 100)
b = res.params.values[1:]
lo = Xtr.quantile(0.01).values; hi = Xtr.quantile(0.99).values
contrib_range = np.abs(b) * (hi - lo)                    # her değişkenin doğrusal öngörücüye maksimum katkısı
scale = 100 / contrib_range.max()
pts = []
for j, f in enumerate(feats):
    grid = np.linspace(lo[j], hi[j], 11)
    base = lo[j] if b[j] > 0 else hi[j]
    p = np.abs(b[j] * (grid - base)) * scale
    pts.append(pd.DataFrame({"Variable": C.PRETTY.get(f, f), "value": grid, "points": p}))
points_tab = pd.concat(pts)
def total_points(X):
    tp = np.zeros(len(X))
    for j, f in enumerate(feats):
        base = lo[j] if b[j] > 0 else hi[j]
        tp += np.abs(b[j] * (np.clip(X[f].values, lo[j], hi[j]) - base)) * scale
    return tp
# toplam puan → olasılık haritası
tp_tr = total_points(Xtr); lp_tr = np.asarray(Xc.values @ res.params.values)
map_fit = np.polyfit(tp_tr, lp_tr, 1)                     # doğrusal (kesme farkı sabit)
tp_grid = np.linspace(0, tp_tr.max() * 1.05, 12)
prob_map = pd.DataFrame({"total_points": tp_grid, "P(TT)": 1 / (1 + np.exp(-np.polyval(map_fit, tp_grid)))})

# ------------------------------------------------------------------ Performans ve kalibrasyon
p_tr = res.predict(Xc); p_te = res.predict(sm.add_constant(Xte, has_constant="add"))
def hosmer_lemeshow(y, p, g=10):
    q = pd.qcut(p, g, labels=False, duplicates="drop"); h = 0; df_ = 0
    for k in np.unique(q):
        m = q == k; o1 = y[m].sum(); e1 = p[m].sum(); n = m.sum()
        if 0 < e1 < n: h += (o1 - e1) ** 2 / e1 + ((n - o1) - (n - e1)) ** 2 / (n - e1); df_ += 1
    return h, stats.chi2.sf(h, df_ - 2), df_
def calib_slope_intercept(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    m0 = sm.Logit(y, np.ones((len(y), 1)), offset=lp).fit(disp=0)   # eğim 1'e sabit → kesişim (calibration-in-the-large)
    return m.params[1], m.bse[1], m0.params[0], m0.bse[0]
perf = []
thr_y, _, _ = youden_threshold(ytr, p_tr)
for name, y_, p_ in (("Train", ytr, np.asarray(p_tr)), ("Test (hold-out)", yte, np.asarray(p_te))):
    auc, l, h, _ = delong_ci(y_, p_); hl, hlp, g = hosmer_lemeshow(y_, p_); sl, slse, ic, icse = calib_slope_intercept(y_, p_)
    m5 = binary_metrics(y_, p_, 0.5); my = binary_metrics(y_, p_, thr_y)
    perf.append({"Set": name, "n": len(y_), "AUC": auc, "AUC_CI_low": l, "AUC_CI_high": h, "Brier": np.mean((p_ - y_) ** 2),
                 "HL_chi2": hl, "HL_df": g - 2, "HL_p": hlp, "Calib_slope": sl, "Calib_slope_CI": f"{sl-1.96*slse:.2f}–{sl+1.96*slse:.2f}",
                 "Calib_intercept": ic, "Calib_intercept_CI": f"{ic-1.96*icse:.2f}–{ic+1.96*icse:.2f}",
                 "Sens@0.5": m5["Sensitivity"], "Spec@0.5": m5["Specificity"], "Acc@0.5": m5["Accuracy"],
                 f"Youden_thr(train)": thr_y, "Sens@Youden": my["Sensitivity"], "Spec@Youden": my["Specificity"], "Acc@Youden": my["Accuracy"]})
perf = pd.DataFrame(perf); print("\n" + perf.round(3).to_string(index=False))
boot_te = bootstrap_metrics(yte, np.asarray(p_te), 0.5, n_boot=C.N_BOOT, seed=RS)

# ------------------------------------------------------------------ Basit puan skoru (Youden eşikleri; 1 puan/kriter)
crit = []
for j, f in enumerate(feats):
    x = Xtr[f].values; s = x if b[j] > 0 else -x
    t, _, _ = youden_threshold(ytr, s); t = t if b[j] > 0 else -t
    crit.append({"Feature": C.PRETTY.get(f, f), "code": f, "criterion": f"{'>' if b[j] > 0 else '<'} {t:.3g}", "thr": t, "dir": 1 if b[j] > 0 else -1})
crit = pd.DataFrame(crit)
def score(X): return sum(((X[c.code].values > c.thr) if c.dir > 0 else (X[c.code].values < c.thr)).astype(int) for c in crit.itertuples())
s_tr, s_te = score(Xtr), score(Xte)
rate = pd.DataFrame({"score": range(len(feats) + 1)})
for name, s_, y_ in (("train", s_tr, ytr), ("test", s_te, yte)):
    rate[f"n_{name}"] = [int((s_ == k).sum()) for k in rate.score]
    rate[f"TT_rate_{name}"] = [y_[s_ == k].mean() if (s_ == k).any() else np.nan for k in rate.score]
rate["AUC_score_train"] = roc_auc_score(ytr, s_tr); rate["AUC_score_test"] = roc_auc_score(yte, s_te)
print("\nBasit skor:\n" + crit[["Feature", "criterion"]].to_string(index=False)); print(rate.round(3).to_string(index=False))

# ------------------------------------------------------------------ Kaydet
with pd.ExcelWriter(OUT / "Table13_nomogram_logistic.xlsx") as xw:
    coef.to_excel(xw, sheet_name="coefficients", index=False); points_tab.to_excel(xw, sheet_name="points_0_100", index=False)
    prob_map.to_excel(xw, sheet_name="total_points_to_prob", index=False)
with pd.ExcelWriter(OUT / "Table14_nomogram_performance.xlsx") as xw:
    perf.to_excel(xw, sheet_name="performance", index=False); boot_te.to_excel(xw, sheet_name="test_bootstrap_0.5")
with pd.ExcelWriter(OUT / "Table15_simple_score.xlsx") as xw:
    crit.to_excel(xw, sheet_name="criteria", index=False); rate.to_excel(xw, sheet_name="rates", index=False)
json.dump({"features": feats, "pretty": [C.PRETTY.get(f, f) for f in feats], "intercept": float(res.params.values[0]),
           "beta": b.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "scale": float(scale), "points_to_lp": map_fit.tolist(),
           "train_median": Xtr.median().tolist(), "youden_threshold": float(thr_y),
           "simple_score": crit[["code", "thr", "dir", "criterion"]].to_dict("records")},
          open(OUT / "nomogram_model.json", "w"), indent=2)

# ------------------------------------------------------------------ Şekiller
k = len(feats)
fig, axes = plt.subplots(k + 2, 1, figsize=(9.5, 1.0 * (k + 2) + 0.6))
ax = axes[0]; ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_ylabel("Points", rotation=0, ha="right", va="center"); ax.xaxis.tick_top(); ax.set_xticks(range(0, 101, 10))
for j, f in enumerate(feats):
    ax = axes[j + 1]; ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_ylabel(C.PRETTY.get(f, f), rotation=0, ha="right", va="center", fontsize=9)
    base = lo[j] if b[j] > 0 else hi[j]; pmax = abs(b[j] * (hi[j] - lo[j])) * scale
    n_t = int(max(2, min(8, pmax // 10 + 2)))                          # kısa eksende az etiket
    grid = np.linspace(lo[j], hi[j], n_t); pnt = np.abs(b[j] * (grid - base)) * scale
    ax.hlines(0, 0, pmax, color="k", lw=1.2); ax.set_xticks(pnt); ax.set_xticklabels([f"{v:.3g}" for v in grid], fontsize=8); ax.set_ylim(-1, 1)
    ax.text(pmax + 1.5, 0, f"OR {np.exp(b[j]):.2f}", va="center", fontsize=7, color="#555")
ax = axes[-1]; ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_ylabel("Total points\n(P(TT))", rotation=0, ha="right", va="center", fontsize=9)
tp_max = float(tp_tr.max()); probs = np.array([0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
lp_at = np.log(probs / (1 - probs)); tp_at = (lp_at - map_fit[1]) / map_fit[0]
keep = (tp_at >= 0) & (tp_at <= tp_max); tp_at, probs = tp_at[keep], probs[keep]
ax.hlines(0, 0, 100, color="k", lw=1.2); ax.set_xticks(tp_at / tp_max * 100)
ax.set_xticklabels([f"{t:.0f}\n({p:.2f})" for t, p in zip(tp_at, probs)], fontsize=8); ax.set_ylim(-1, 1)
ax.text(0, -0.9, f"0 pts", fontsize=7, color="#555"); ax.text(99, -0.9, f"{tp_max:.0f} pts", fontsize=7, color="#555", ha="right")
for a in axes: a.grid(False); [sp.set_visible(False) for sp in a.spines.values()]
fig.suptitle("Nomogram for testicular torsion vs epididymo-orchitis (logistic regression)", y=0.995)
plt.tight_layout(); plt.savefig(OUT / "Fig_nomogram.png", dpi=300); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, (name, y_, p_) in zip(axes, (("Train", ytr, np.asarray(p_tr)), ("Test (hold-out)", yte, np.asarray(p_te)))):
    q = pd.qcut(p_, 8, labels=False, duplicates="drop"); obs = [y_[q == k].mean() for k in np.unique(q)]; pr = [p_[q == k].mean() for k in np.unique(q)]
    ax.plot(pr, obs, "o-", label="Nomogram"); ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
    r = perf[perf.Set == name].iloc[0]
    ax.set_title(f"{name}: AUC {r.AUC:.3f}, HL p={r.HL_p:.3f}, slope {r.Calib_slope:.2f}, Brier {r.Brier:.3f}", fontsize=9)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed TT frequency"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "Fig_nomogram_calibration.png", dpi=300); plt.close()

fig, ax = plt.subplots(figsize=(5.5, 3.8))
w = 0.38; x = np.arange(len(rate))
ax.bar(x - w / 2, rate.TT_rate_train * 100, w, label="Train"); ax.bar(x + w / 2, rate.TT_rate_test * 100, w, label="Test")
for i, r in rate.iterrows(): ax.text(i, 2, f"n={r.n_train}/{r.n_test}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xlabel("Simple score (number of criteria met)"); ax.set_ylabel("Observed TT rate (%)"); ax.legend(); ax.set_title("Simple bedside score")
plt.tight_layout(); plt.savefig(OUT / "Fig_score_TT_rate.png", dpi=300); plt.close()
print(f"\nBitti → {OUT}")
