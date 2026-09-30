# -*- coding: utf-8 -*-
"""
ADIM 4 — Öznitelik seçimi: hibrit ikili PSO-GWO + karşılaştırma yöntemleri (YALNIZ EĞİTİM SETİ).

Uygunluk (fitness): dengeli Random Forest (class_weight="balanced", 100 ağaç, max_depth=5) ile 5-kat tabakalı CV
ortalama MCC − λ·(seçilen/toplam)  [λ=0.02: eşit performansta daha az öznitelik tercih edilir].

Yöntemler: PSO-GWO (hibrit), PSO, GWO, LASSO, ElasticNet, RF-önem, RFE, ANOVA-F, Mutual Information, Chi-kare,
Boruta (paket kuruluysa). Her alt küme aynı 5×5 tekrarlı CV ile değerlendirilir; kararlılık için PSO-GWO ve en
iyi rakip N_SPLITS rastgele bölmede karşılaştırılır (Wilcoxon).

Kullanım:
  python 04_feature_selection.py [--drop-age]
Sonra seçilen küme ile modelleme:
  python 02_nested_cv_models.py --features-json outputs/04_feature_selection/selected_features.json --fs-method psogwo
  python 03_xai.py --tag _fs_psogwo

Çıktılar (OUT_DIR/04_feature_selection[_noage]/): Table11_feature_selection_comparison.xlsx, selected_features.json,
  Table12_stability_psogwo_vs_best.xlsx, Fig_convergence.png, Fig_feature_selection_frequency.png
"""
import argparse, json, time, warnings, os
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.pipeline import make_pipeline
import config as C

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper")
ap = argparse.ArgumentParser()
ap.add_argument("--drop-age", action="store_true")
ap.add_argument("--feature-set", default=C.FEATURE_SET)
ap.add_argument("--particles", type=int, default=30)
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--alpha", type=float, default=0.5, help="hibritleşme katsayısı (0=saf GWO, 1=saf PSO)")
ap.add_argument("--n-splits", type=int, default=20, help="kararlılık analizi rastgele bölme sayısı")
args = ap.parse_args()
TAG = "_noage" if args.drop_age else ""
OUT = C.out(f"04_feature_selection{TAG}")
RS = C.RANDOM_STATE
if C.QUICK:
    args.particles, args.iters, args.n_splits = 6, 3, 3

df = pd.read_csv(C.OUT_DIR / "00_data" / "clean_data.csv")
split = pd.read_csv(C.OUT_DIR / "00_data" / "split_indices.csv")
feats = [c for c in C.FEATURE_SETS[args.feature_set] if c in df.columns and not (args.drop_age and c == "Age")]
tr = split.loc[split.split == "train", "row_id"].values
X = df.loc[tr, feats].values.astype(float); y = df.loc[tr, C.TARGET].values
D = len(feats)
print(f"Eğitim seti n={len(y)} (TT={y.sum()}), aday öznitelik={D}")

mcc = make_scorer(matthews_corrcoef)
def base_rf(seed=RS):
    return RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=seed, n_jobs=C.N_JOBS)

_cache = {}
def fitness(mask, lam=0.02, seed=RS):
    key = tuple(mask.astype(int))
    if key in _cache: return _cache[key]
    if mask.sum() == 0: return -1.0
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    sc = cross_val_score(base_rf(seed), X[:, mask.astype(bool)], y, cv=cv, scoring=mcc).mean()
    val = sc - lam * mask.sum() / D
    _cache[key] = val
    return val

def sigmoid(v): return 1 / (1 + np.exp(-v))

# ------------------------------------------------------------------ Hibrit ikili PSO-GWO
def pso_gwo(n=30, iters=50, alpha=0.5, seed=RS, log=None):
    """alpha: PSO hız bileşeninin ağırlığı; (1-alpha): GWO (alfa/beta/delta kurtlara yaklaşma) bileşeni."""
    rng = np.random.default_rng(seed)
    pos = (rng.random((n, D)) < 0.5).astype(int)
    for i in range(n):
        if pos[i].sum() == 0: pos[i, rng.integers(D)] = 1
    vel = rng.uniform(-1, 1, (n, D))
    fit = np.array([fitness(p) for p in pos])
    pbest, pbest_f = pos.copy(), fit.copy()
    hist = []
    for t in range(iters):
        order = np.argsort(-fit)
        a_w, b_w, d_w = pos[order[0]], pos[order[1]], pos[order[2]]
        gbest, gbest_f = pbest[np.argmax(pbest_f)], pbest_f.max()
        a = 2 - 2 * t / iters                         # GWO: 2→0
        w = 0.9 - 0.5 * t / iters                     # PSO atalet
        for i in range(n):
            r1, r2 = rng.random(D), rng.random(D)
            v_pso = w * vel[i] + 2.0 * r1 * (pbest[i] - pos[i]) + 2.0 * r2 * (gbest - pos[i])
            # GWO: üç lider kurta göre sürekli konum, ikilileştirme için sigmoid
            x_gwo = np.zeros(D)
            for leader in (a_w, b_w, d_w):
                A = 2 * a * rng.random(D) - a; Cc = 2 * rng.random(D)
                x_gwo += leader - A * np.abs(Cc * leader - pos[i])
            x_gwo /= 3
            vel[i] = alpha * v_pso + (1 - alpha) * (x_gwo - pos[i])
            vel[i] = np.clip(vel[i], -6, 6)
            new = (rng.random(D) < sigmoid(vel[i])).astype(int)
            if new.sum() == 0: new[rng.integers(D)] = 1
            pos[i] = new
            fit[i] = fitness(pos[i])
            if fit[i] > pbest_f[i]: pbest[i], pbest_f[i] = pos[i].copy(), fit[i]
        hist.append(pbest_f.max())
        if log is not None: log(f"  iter {t+1:3d}/{iters}  best={pbest_f.max():.4f}  n_feat={pbest[np.argmax(pbest_f)].sum()}")
    b = pbest[np.argmax(pbest_f)].astype(bool)
    return b, pbest_f.max(), hist

# ------------------------------------------------------------------ Karşılaştırma yöntemleri
def sel_lasso(Xs, y, seed):
    m = LogisticRegression(penalty="l1", solver="saga", C=0.3, max_iter=5000, class_weight="balanced", random_state=seed).fit(Xs, y)
    return np.abs(m.coef_[0]) > 1e-6
def sel_enet(Xs, y, seed):
    m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.3, max_iter=5000, class_weight="balanced", random_state=seed).fit(Xs, y)
    return np.abs(m.coef_[0]) > 1e-6
def topk(scores, k): 
    m = np.zeros(D, bool); m[np.argsort(-scores)[:k]] = True; return m
def sel_rf(Xs, y, seed, k): return topk(base_rf(seed).fit(Xs, y).feature_importances_, k)
def sel_rfe(Xs, y, seed, k): return RFE(LogisticRegression(max_iter=3000, class_weight="balanced"), n_features_to_select=k).fit(Xs, y).support_
def sel_anova(Xs, y, seed, k): return topk(f_classif(Xs, y)[0], k)
def sel_mi(Xs, y, seed, k): return topk(mutual_info_classif(Xs, y, random_state=seed), k)
def sel_chi2(Xs, y, seed, k): return topk(chi2(MinMaxScaler().fit_transform(Xs), y)[0], k)
def sel_boruta(Xs, y, seed, k=None):
    try:
        from boruta import BorutaPy
        b = BorutaPy(RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=5, n_jobs=C.N_JOBS, random_state=seed), n_estimators="auto", random_state=seed, max_iter=50).fit(Xs, y)
        return b.support_ | b.support_weak_
    except ImportError:
        return None

def run_all(seed, verbose=False, light=False, only=None):
    """light=True: kararlılık analizi için küçültülmüş PSO-GWO (yarı parçacık/iterasyon), PSO/GWO/Boruta atlanır.
    only: yalnız bu yöntemler hesaplanır (ör. ["PSO-GWO", "LASSO"])."""
    Xs = StandardScaler().fit_transform(X)
    out = {}
    lg = print if verbose else None
    div = 2 if light else 1
    m, f, hist = pso_gwo(max(6, args.particles // div), max(10, args.iters // div), args.alpha, seed, lg); out["PSO-GWO"] = (m, hist)
    k = int(m.sum())                                  # rakipler aynı boyutta küme seçer (adil kıyas)
    if not light:
        m2, _, h2 = pso_gwo(args.particles, args.iters, 1.0, seed); out["PSO"] = (m2, h2)
        m3, _, h3 = pso_gwo(args.particles, args.iters, 0.0, seed); out["GWO"] = (m3, h3)
    out["LASSO"] = (sel_lasso(Xs, y, seed), None); out["ElasticNet"] = (sel_enet(Xs, y, seed), None)
    out["RF importance"] = (sel_rf(Xs, y, seed, k), None); out["RFE"] = (sel_rfe(Xs, y, seed, k), None)
    out["ANOVA F-test"] = (sel_anova(Xs, y, seed, k), None); out["Mutual Info"] = (sel_mi(Xs, y, seed, k), None)
    out["Chi-square"] = (sel_chi2(Xs, y, seed, k), None)
    b = None if light else sel_boruta(Xs, y, seed)
    if b is not None: out["Boruta"] = (b, None)
    out["All features"] = (np.ones(D, bool), None)
    if only is not None:
        out = {kk: vv for kk, vv in out.items() if kk in only}
    return out

# ------------------------------------------------------------------ Ana çalıştırma
t0 = time.time()
print("PSO-GWO çalışıyor…")
res = run_all(RS, verbose=True)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1 if C.QUICK else 5, random_state=RS)
rows = []
for name, (mask, hist) in res.items():
    if mask.sum() == 0: continue
    sc = cross_val_score(base_rf(), X[:, mask], y, cv=rcv, scoring=mcc)
    auc = cross_val_score(base_rf(), X[:, mask], y, cv=rcv, scoring="roc_auc")
    rows.append({"Method": name, "MCC_mean": sc.mean(), "MCC_SD": sc.std(ddof=1), "AUC_mean": auc.mean(), "AUC_SD": auc.std(ddof=1),
                 "N_features": int(mask.sum()), "Selected": ", ".join(np.array(feats)[mask])})
tab = pd.DataFrame(rows).sort_values("MCC_mean", ascending=False)
tab.to_excel(OUT / "Table11_feature_selection_comparison.xlsx", index=False)
print("\n" + tab[["Method", "MCC_mean", "MCC_SD", "AUC_mean", "N_features", "Selected"]].round(3).to_string(index=False))

# Yakınsama grafiği
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for name in ("PSO-GWO", "PSO", "GWO"):
    ax.plot(range(1, len(res[name][1]) + 1), res[name][1], lw=1.8, label=name)
ax.set_xlabel("Iteration"); ax.set_ylabel("Best fitness (CV MCC − penalty)"); ax.legend(); ax.set_title("Convergence")
plt.tight_layout(); plt.savefig(OUT / "Fig_convergence.png", dpi=300); plt.close()

# Kararlılık: PSO-GWO vs en iyi rakip, N rastgele alt-bölmede (eğitim seti içinde %80 alt örnek)
best_rival = tab[~tab.Method.isin(["PSO-GWO", "PSO", "GWO", "All features"])].iloc[0]["Method"]
print(f"\nKararlılık analizi: PSO-GWO vs {best_rival} ({args.n_splits} bölme)…")
rng = np.random.default_rng(RS); stab = []; freq = np.zeros(D)
X_full, y_full = X.copy(), y.copy()
t_st = time.time()
for s in range(args.n_splits):
    idx = rng.choice(len(y_full), int(0.8 * len(y_full)), replace=False)
    X, y = X_full[idx], y_full[idx]; _cache.clear()
    r = run_all(RS + 100 + s, light=True, only=["PSO-GWO", best_rival])
    m_a = r["PSO-GWO"][0]; m_b = r[best_rival][0]; freq += m_a
    cv = StratifiedKFold(5, shuffle=True, random_state=s)
    stab.append({"split": s, "PSO-GWO": cross_val_score(base_rf(), X[:, m_a], y, cv=cv, scoring=mcc).mean(),
                 best_rival: cross_val_score(base_rf(), X[:, m_b], y, cv=cv, scoring=mcc).mean(), "n_psogwo": int(m_a.sum()),
                 "selected": ", ".join(np.array(feats)[m_a])})
    pd.DataFrame(stab).to_csv(OUT / "stability_per_split_partial.csv", index=False)
    el = (time.time() - t_st) / 60
    print(f"  bölme {s+1:2d}/{args.n_splits}  PSO-GWO={stab[-1]['PSO-GWO']:.3f}  {best_rival}={stab[-1][best_rival]:.3f}  "
          f"geçen {el:.1f} dk, tahmini kalan {el/(s+1)*(args.n_splits-s-1):.1f} dk", flush=True)
X, y = X_full, y_full
stab = pd.DataFrame(stab)
w = stats.wilcoxon(stab["PSO-GWO"], stab[best_rival]) if len(stab) >= 5 else None
summary = pd.DataFrame({"Method": ["PSO-GWO", best_rival], "MCC_mean": [stab["PSO-GWO"].mean(), stab[best_rival].mean()],
                        "MCC_SD": [stab["PSO-GWO"].std(), stab[best_rival].std()], "Wilcoxon_p": [w.pvalue if w else np.nan] * 2})
with pd.ExcelWriter(OUT / "Table12_stability_psogwo_vs_best.xlsx") as xw:
    summary.to_excel(xw, sheet_name="summary", index=False); stab.to_excel(xw, sheet_name="per_split", index=False)
    pd.DataFrame({"Feature": feats, "selection_frequency": freq / args.n_splits}).sort_values("selection_frequency", ascending=False).to_excel(xw, sheet_name="selection_frequency", index=False)
print(summary.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(6, 4)); o = np.argsort(-freq)
ax.barh(np.array(feats)[o][::-1], (freq / args.n_splits)[o][::-1], color="#fc8d62"); ax.set_xlabel(f"Selection frequency over {args.n_splits} resamples")
ax.set_title("PSO-GWO feature selection stability"); plt.tight_layout(); plt.savefig(OUT / "Fig_feature_selection_frequency.png", dpi=300); plt.close()

sel = {name: [str(f) for f in np.array(feats)[mask]] for name, (mask, _) in res.items()}
sel["_meta"] = {"feature_set": args.feature_set, "drop_age": args.drop_age, "particles": args.particles, "iters": args.iters, "alpha": args.alpha}
json.dump(sel, open(OUT / "selected_features.json", "w"), indent=2, ensure_ascii=False)
print(f"\nPSO-GWO seçimi: {sel['PSO-GWO']}\nSüre {(time.time()-t0)/60:.1f} dk → {OUT}")
