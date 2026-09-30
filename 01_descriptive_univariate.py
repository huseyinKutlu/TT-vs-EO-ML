# -*- coding: utf-8 -*-
"""
ADIM 1 — Tanımlayıcı istatistikler ve tek değişkenli (univariate) analiz.

Tüm kohort (n=526) üzerinde raporlanır; ayrıca train/test karşılaştırma tablosu üretilir
(bölmenin dengeli olduğunu göstermek için).

Çıktılar (OUT_DIR/01_univariate/):
  Table1_descriptives_by_group.xlsx : mean±SD, median (Q1–Q3), Shapiro, test, p, p_FDR, Cliff δ, Hedges g, bootstrap CI
  Table2_univariate_ROC.xlsx        : AUC (DeLong CI), Youden cut-off, Se/Sp/PPV/NPV/Acc/LR±
  Table3_univariate_logistic.xlsx   : değişken başına OR (per 1 SD) + ham birim OR
  TableS_train_vs_test.xlsx         : bölme dengesi
  Figures: boxplots, univariate ROC panel, Spearman heatmap
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.metrics import roc_curve
import config as C
from utils import (cliffs_delta, cliffs_label, hedges_g, bootstrap_median_diff, delong_ci, youden_threshold,
                   binary_metrics, fmt_p)

OUT = C.out("01_univariate")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

df = pd.read_csv(C.OUT_DIR / "00_data" / "clean_data.csv")
split = pd.read_csv(C.OUT_DIR / "00_data" / "split_indices.csv")
feats = [c for c in C.FEATURE_SETS["all"] if c in df.columns]
y = df[C.TARGET].values
g1 = df[y == 1]; g0 = df[y == 0]
print(f"n={len(df)}  {C.POS_LABEL_NAME}={len(g1)}  {C.NEG_LABEL_NAME}={len(g0)}")

# ------------------------------------------------------------------ Tablo 1: grup karşılaştırması
rows = []
for c in feats:
    a, b = g1[c].dropna().values, g0[c].dropna().values
    sw_all = stats.shapiro(df[c].sample(min(len(df), 5000), random_state=0)).pvalue
    sw1, sw0 = stats.shapiro(a).pvalue, stats.shapiro(b).pvalue
    normal = (sw1 > 0.05) and (sw0 > 0.05)
    lev_p = stats.levene(a, b).pvalue
    if normal:
        t, p = stats.ttest_ind(a, b, equal_var=lev_p > 0.05)
        test = "Student t" if lev_p > 0.05 else "Welch t"
        stat = t
    else:
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        stat, p, test = u.statistic, u.pvalue, "Mann–Whitney U"
    md, lo, hi = bootstrap_median_diff(a, b, n_boot=C.N_BOOT, seed=C.RANDOM_STATE)
    d = cliffs_delta(a, b)
    rows.append({
        "Variable": C.PRETTY.get(c, c), "code": c,
        f"{C.POS_LABEL_NAME} mean±SD": f"{a.mean():.2f} ± {a.std(ddof=1):.2f}",
        f"{C.NEG_LABEL_NAME} mean±SD": f"{b.mean():.2f} ± {b.std(ddof=1):.2f}",
        f"{C.POS_LABEL_NAME} median (Q1–Q3)": f"{np.median(a):.2f} ({np.percentile(a,25):.2f}–{np.percentile(a,75):.2f})",
        f"{C.NEG_LABEL_NAME} median (Q1–Q3)": f"{np.median(b):.2f} ({np.percentile(b,25):.2f}–{np.percentile(b,75):.2f})",
        "Shapiro p (TT)": sw1, "Shapiro p (EO)": sw0, "Levene p": lev_p, "Test": test, "Statistic": stat,
        "p": p, "Median diff (TT−EO)": md, "Boot 95% CI low": lo, "Boot 95% CI high": hi,
        "Cliff's δ": d, "Effect": cliffs_label(d), "Hedges g": hedges_g(a, b),
    })
t1 = pd.DataFrame(rows)
t1["p_FDR"] = multipletests(t1["p"], method="fdr_bh")[1]
t1["p (fmt)"] = t1["p"].map(fmt_p); t1["p_FDR (fmt)"] = t1["p_FDR"].map(fmt_p)
t1 = t1.sort_values("p")
t1.to_excel(OUT / "Table1_descriptives_by_group.xlsx", index=False)
print("\nTABLO 1 (özet):")
print(t1[["Variable", "Test", "p (fmt)", "p_FDR (fmt)", "Cliff's δ", "Effect"]].to_string(index=False))

# Genel tanımlayıcılar (tüm kohort)
desc = df[feats].describe(percentiles=[.25, .5, .75]).T
desc["skew"] = df[feats].skew(); desc["kurtosis"] = df[feats].kurtosis()
desc.index = [C.PRETTY.get(c, c) for c in desc.index]
desc.to_excel(OUT / "Table0_overall_descriptives.xlsx")

# ------------------------------------------------------------------ Tablo 2: tek değişkenli ROC
rows = []
for c in feats:
    x = df[c].values
    auc, lo, hi, se = delong_ci(y, x)
    direction = ">" if auc >= 0.5 else "<"
    score = x if auc >= 0.5 else -x
    if auc < 0.5:
        auc, lo, hi = 1 - auc, 1 - hi, 1 - lo
    thr, se_, sp_ = youden_threshold(y, score)
    thr_orig = thr if direction == ">" else -thr
    m = binary_metrics(y, score, thr)
    z = (auc - 0.5) / se; p_auc = 2 * stats.norm.sf(abs(z))
    rows.append({"Variable": C.PRETTY.get(c, c), "code": c, "AUC": auc, "AUC 95% CI low": lo, "AUC 95% CI high": hi,
                 "p (AUC vs 0.5)": p_auc, "Direction (TT)": direction, "Youden cut-off": thr_orig,
                 "Sensitivity": m["Sensitivity"], "Specificity": m["Specificity"], "PPV": m["PPV"], "NPV": m["NPV"],
                 "Accuracy": m["Accuracy"], "LR+": m["LR+"], "LR-": m["LR-"], "Youden J": m["Sensitivity"] + m["Specificity"] - 1})
t2 = pd.DataFrame(rows).sort_values("AUC", ascending=False)
t2["p (fmt)"] = t2["p (AUC vs 0.5)"].map(fmt_p)
t2.to_excel(OUT / "Table2_univariate_ROC.xlsx", index=False)
print("\nTABLO 2 (tek değişkenli ROC):")
print(t2[["Variable", "AUC", "AUC 95% CI low", "AUC 95% CI high", "Youden cut-off", "Sensitivity", "Specificity"]].round(3).to_string(index=False))

# ------------------------------------------------------------------ Tablo 3: tek değişkenli lojistik regresyon
rows = []
for c in feats:
    x = df[c].values.astype(float)
    xs = (x - x.mean()) / x.std(ddof=1)
    for label, xx in (("per 1 SD", xs), ("per unit", x)):
        try:
            res = sm.Logit(y, sm.add_constant(xx)).fit(disp=0, maxiter=200)
            b, se = res.params[1], res.bse[1]
            rows.append({"Variable": C.PRETTY.get(c, c), "code": c, "Scale": label, "OR": np.exp(b),
                         "OR 95% CI low": np.exp(b - 1.96 * se), "OR 95% CI high": np.exp(b + 1.96 * se),
                         "p": res.pvalues[1], "McFadden R2": res.prsquared})
        except Exception as e:
            rows.append({"Variable": C.PRETTY.get(c, c), "code": c, "Scale": label, "OR": np.nan, "p": np.nan, "note": str(e)})
t3 = pd.DataFrame(rows)
t3.loc[t3.Scale == "per 1 SD", "p_FDR"] = multipletests(t3.loc[t3.Scale == "per 1 SD", "p"].fillna(1), method="fdr_bh")[1]
t3.to_excel(OUT / "Table3_univariate_logistic.xlsx", index=False)

# ------------------------------------------------------------------ Tablo S: train vs test dengesi
d2 = df.merge(split, left_index=True, right_on="row_id")
rows = []
for c in feats + [C.TARGET]:
    a, b = d2.loc[d2.split == "train", c], d2.loc[d2.split == "test", c]
    if c == C.TARGET:
        p = stats.chi2_contingency(pd.crosstab(d2.split, d2[c]))[1]
        rows.append({"Variable": "TT prevalence", "Train": f"{a.mean()*100:.1f}%", "Test": f"{b.mean()*100:.1f}%", "p": p})
    else:
        p = stats.mannwhitneyu(a, b).pvalue
        rows.append({"Variable": C.PRETTY.get(c, c),
                     "Train": f"{a.median():.2f} ({a.quantile(.25):.2f}–{a.quantile(.75):.2f})",
                     "Test": f"{b.median():.2f} ({b.quantile(.25):.2f}–{b.quantile(.75):.2f})", "p": p})
pd.DataFrame(rows).to_excel(OUT / "TableS_train_vs_test.xlsx", index=False)

# ------------------------------------------------------------------ Şekiller
# Boxplot paneli
n = len(feats); ncol = 5; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow))
pal = {0: "#4C72B0", 1: "#C44E52"}
for ax, c in zip(axes.ravel(), feats):
    sns.boxplot(data=df, x=C.TARGET, y=c, hue=C.TARGET, palette=pal, ax=ax, width=0.55, fliersize=2, legend=False)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["EO", "TT"]); ax.set_xlabel("")
    ax.set_ylabel(C.PRETTY.get(c, c), fontsize=9)
    pf = t1.set_index("code").loc[c, "p_FDR"]
    ax.set_title(f"p_FDR = {fmt_p(pf)}", fontsize=9)
    if df[c].skew() > 2: ax.set_yscale("log")
for ax in axes.ravel()[n:]: ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "Fig_boxplots_by_group.png", dpi=300); plt.savefig(OUT / "Fig_boxplots_by_group.pdf"); plt.close()

# Tek değişkenli ROC paneli (en iyi 6 + Age)
top = t2.head(6)["code"].tolist()
fig, ax = plt.subplots(figsize=(5.5, 5.2))
for c in top:
    x = df[c].values; auc = t2.set_index("code").loc[c, "AUC"]; d = t2.set_index("code").loc[c, "Direction (TT)"]
    fpr, tpr, _ = roc_curve(y, x if d == ">" else -x)
    ax.plot(fpr, tpr, lw=1.6, label=f"{c} (AUC {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("Univariate ROC curves — TT vs EO"); ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "Fig_univariate_ROC.png", dpi=300); plt.savefig(OUT / "Fig_univariate_ROC.pdf"); plt.close()

# Spearman korelasyon
corr = df[feats].corr("spearman")
fig, ax = plt.subplots(figsize=(9, 7.5))
mask = np.triu(np.ones_like(corr, bool), 1)
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 6.5},
            square=True, cbar_kws={"shrink": .7}, ax=ax)
ax.set_title("Spearman correlation among candidate predictors")
plt.tight_layout(); plt.savefig(OUT / "Fig_spearman_heatmap.png", dpi=300); plt.close()

# Yaş dağılımı (hakem sorusu için)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
sns.histplot(data=df, x="Age", hue=C.TARGET, bins=30, palette=pal, element="step", stat="density", common_norm=False, ax=ax)
ax.legend(labels=["TT", "EO"]); ax.set_xlabel("Age (years)"); ax.set_title("Age distribution by diagnosis")
plt.tight_layout(); plt.savefig(OUT / "Fig_age_distribution.png", dpi=300); plt.close()

print(f"\nBitti → {OUT}")
