# -*- coding: utf-8 -*-
"""Ortak yardımcı fonksiyonlar: metrikler, DeLong, bootstrap, etki büyüklükleri."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef, f1_score,
                             balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_curve)


# ------------------------------------------------------------------ Temel metrikler
def binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    out = {
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "AUPRC": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Sensitivity": sens, "Specificity": spec, "PPV": ppv, "NPV": npv,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob) if (y_prob.min() >= 0 and y_prob.max() <= 1) else np.nan,
        "LR+": sens / (1 - spec) if spec is not np.nan and spec < 1 else np.inf,
        "LR-": (1 - sens) / spec if spec else np.inf,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn, "Threshold": threshold,
    }
    return out


def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i]), float(tpr[i]), float(1 - fpr[i])


# ------------------------------------------------------------------ Bootstrap CI
def bootstrap_metrics(y_true, y_prob, threshold=0.5, n_boot=2000, seed=42, stratified=True):
    """Tabakalı bootstrap ile metriklerin %95 GA'sı (percentile)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob)
    idx_pos = np.where(y_true == 1)[0]; idx_neg = np.where(y_true == 0)[0]
    rows = []
    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate([rng.choice(idx_pos, len(idx_pos), True), rng.choice(idx_neg, len(idx_neg), True)])
        else:
            idx = rng.choice(len(y_true), len(y_true), True)
        rows.append(binary_metrics(y_true[idx], y_prob[idx], threshold))
    boot = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    point = pd.Series(binary_metrics(y_true, y_prob, threshold))
    res = pd.DataFrame({"Estimate": point, "CI_low": boot.quantile(0.025), "CI_high": boot.quantile(0.975)})
    return res


# ------------------------------------------------------------------ DeLong (Sun & Xu, 2014)
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T + 1
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count; n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]; neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :]); ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n; v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01); sy = np.cov(v10)
    return aucs, sx / m + sy / n


def delong_ci(y_true, y_prob, alpha=0.95):
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(-y_true); m = int(y_true.sum())
    aucs, cov = _fast_delong(y_prob[order][np.newaxis, :], m)
    auc = aucs[0]; se = np.sqrt(cov if np.ndim(cov) == 0 else cov[0, 0])
    z = stats.norm.ppf(0.5 + alpha / 2)
    return auc, max(0, auc - z * se), min(1, auc + z * se), se


def delong_test(y_true, p1, p2):
    """İki korele ROC eğrisi için DeLong testi. Döndürür: (z, p, auc1, auc2)."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true); m = int(y_true.sum())
    preds = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    l = np.array([[1, -1]])
    var = l @ cov @ l.T
    z = (aucs[0] - aucs[1]) / np.sqrt(var[0, 0]) if var[0, 0] > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return float(z), float(p), float(aucs[0]), float(aucs[1])


# ------------------------------------------------------------------ Etki büyüklükleri
def cliffs_delta(a, b):
    """Cliff's delta: P(a>b) - P(a<b); MWU üzerinden hızlı hesap."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def cliffs_label(d):
    d = abs(d)
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def hedges_g(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    return d * J


def bootstrap_median_diff(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    diffs = [np.median(rng.choice(a, len(a), True)) - np.median(rng.choice(b, len(b), True)) for _ in range(n_boot)]
    return np.median(a) - np.median(b), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def fmt_p(p):
    if pd.isna(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_ci(est, lo, hi, nd=3):
    return f"{est:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def decision_curve(y_true, y_prob, thresholds=np.linspace(0.01, 0.99, 99)):
    """Net benefit (Vickers & Elkin) — model, treat-all, treat-none."""
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob)
    n = len(y_true); prev = y_true.mean()
    rows = []
    for t in thresholds:
        pred = y_prob >= t
        tp = np.sum(pred & (y_true == 1)); fp = np.sum(pred & (y_true == 0))
        nb = tp / n - fp / n * (t / (1 - t))
        nb_all = prev - (1 - prev) * (t / (1 - t))
        rows.append({"threshold": t, "NB_model": nb, "NB_treat_all": nb_all, "NB_treat_none": 0.0})
    return pd.DataFrame(rows)
