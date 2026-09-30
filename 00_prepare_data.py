# -*- coding: utf-8 -*-
"""
ADIM 0 — Veri temizleme, sızıntı (leakage) denetimi ve sabit tabakalı bölme.

Çıktılar (OUT_DIR/00_data/):
  clean_data.csv            : temizlenmiş tam veri (kısa sütun adları, TT hedefi)
  split_indices.csv         : her hastanın train/test etiketi (tüm adımlar aynı bölmeyi kullanır)
  leakage_audit.xlsx        : sızıntı/kolinearite denetim tabloları
  data_cleaning_log.txt     : yapılan tüm düzeltmelerin kaydı
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import config as C

OUT = C.out("00_data")
log = []
def L(msg):
    print(msg); log.append(msg)

# ------------------------------------------------------------------ 1) Oku
raw = pd.read_excel(C.DATA_PATH, sheet_name=C.SHEET_NAME)
L(f"Ham veri: {raw.shape[0]} satır × {raw.shape[1]} sütun")
L(f"Sütunlar: {list(raw.columns)}")

# ------------------------------------------------------------------ 2) Boş sütunları at, yeniden adlandır
for c in C.DROP_COLS_RAW:
    if c in raw.columns:
        L(f"Sütun atıldı (tamamen boş / kopya): '{c}' — dolu hücre sayısı = {raw[c].notna().sum()}")
        raw = raw.drop(columns=c)
raw.columns = [str(c).strip() for c in raw.columns]
raw = raw.rename(columns={k.strip(): v for k, v in C.RENAME_MAP.items()})
raw = raw.rename(columns={C.TARGET_COL_RAW: C.TARGET})
raw.columns = [str(c).strip() for c in raw.columns]

# ------------------------------------------------------------------ 3) Sayısallaştır (virgül ondalık, fazla karakter)
def to_num(series: pd.Series, name: str):
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[^\d,.\-eE+]", "", regex=True)      # yalnız rakam ve ayırıcı kalsın
    s = s.str.rstrip(",.")                                  # "10,01," gibi sondaki ayırıcı
    s = s.str.replace(",", ".", regex=False)
    s = s.apply(lambda v: v if v.count(".") <= 1 else v.replace(".", "", v.count(".") - 1))
    num = pd.to_numeric(s, errors="coerce")
    bad = series[num.isna() & series.notna()]
    if len(bad):
        L(f"  [{name}] sayıya çevrilemeyen {len(bad)} hücre: {bad.unique()[:10]}")
    return num

df = raw.copy()
for c in df.columns:
    if c == C.TARGET:
        continue
    if df[c].dtype == object:
        n_comma = df[c].astype(str).str.contains(",").sum()
        L(f"Sütun '{c}': metin tipi; virgüllü hücre = {n_comma} → sayıya çevrildi")
    df[c] = to_num(df[c], c)
df[C.TARGET] = df[C.TARGET].astype(int)

# ------------------------------------------------------------------ 4) Eksik / kopya / aykırı
na = df.isna().sum()
L(f"Eksik değer (temizleme sonrası): {na[na > 0].to_dict() if na.sum() else 'yok'}")
dup = df.duplicated().sum()
L(f"Tam kopya satır: {dup}")
if dup:
    df = df.drop_duplicates().reset_index(drop=True)
    L("  → kopya satırlar çıkarıldı")

# Fizyolojik sınır kontrolü (yalnızca uyarı, otomatik silme yok)
plaus = {"Age": (0, 110), "WBC": (0.5, 100), "RDW": (8, 40), "PLT": (5, 1500), "PDW": (5, 35), "PCT": (0.01, 1.5),
         "MPV": (4, 20), "NEU": (0.05, 80), "LYM": (0.05, 30), "MONO": (0.01, 10), "LYM_pct": (0, 100), "MONO_pct": (0, 100)}
for c, (lo, hi) in plaus.items():
    if c in df:
        m = (df[c] < lo) | (df[c] > hi)
        if m.any():
            L(f"UYARI [{c}] beklenen aralık dışı ({lo}-{hi}) {m.sum()} değer: {df.loc[m, c].round(3).tolist()[:10]}")

# Türetilmiş oranların tutarlılığı (veri girişi hatası tespiti)
checks = {
    "NLR": df["NEU"] / df["LYM"], "PLR": df["PLT"] / df["LYM"], "MLR": df["MONO"] / df["LYM"],
    "SII": df["PLT"] * df["NEU"] / df["LYM"], "PIV": df["PLT"] * df["NEU"] * df["MONO"] / df["LYM"],
}
for c, recomputed in checks.items():
    if c in df:
        rel = (df[c] - recomputed).abs() / recomputed.abs().clip(lower=1e-9)
        n_bad = (rel > 0.02).sum()
        L(f"Tutarlılık [{c}] yeniden hesap ile >%2 sapma: {n_bad} satır (medyan sapma %{100*rel.median():.3f})")
        if n_bad:
            L(f"   satırlar: {df.index[rel > 0.02].tolist()[:15]}")

# ------------------------------------------------------------------ 5) Sızıntı (leakage) denetimi
feats = [c for c in df.columns if c != C.TARGET]
y = df[C.TARGET].values
audit = []
Xf = df[feats].fillna(df[feats].median())
mi = mutual_info_classif(Xf, y, random_state=C.RANDOM_STATE)
for c, m in zip(feats, mi):
    x = df[c]
    auc = roc_auc_score(y, x.fillna(x.median()))
    auc = max(auc, 1 - auc)
    # sınıf içi sabitlik: bir sınıfta tek değer alan sütun → sızıntı şüphesi
    const_in_class = any(x[y == k].nunique() <= 1 for k in (0, 1))
    # mükemmel ayrım: tek eşik ile %100 doğruluk
    perfect = auc >= 0.999
    flag = []
    if perfect or const_in_class: flag.append("!!! MUHTEMEL SIZINTI — çıkarılmalı")
    elif auc >= 0.90: flag.append("çok güçlü tek değişkenli ayrım — klinik olarak makul mü? (Age: yaş dağılımı hastalık tanımının parçası, sızıntı değil; --drop-age duyarlılık analizi önerilir)")
    if x.nunique() <= 1: flag.append("sıfır varyans — çıkarılmalı")
    if x.isna().mean() > 0.3: flag.append("çok eksik")
    audit.append({"Feature": c, "n_unique": x.nunique(), "missing_%": 100 * x.isna().mean(),
                  "univariate_AUC(max)": auc, "mutual_info": m, "const_within_class": const_in_class,
                  "flag": "; ".join(flag)})
audit = pd.DataFrame(audit).sort_values("univariate_AUC(max)", ascending=False)
L("\nSızıntı denetimi (tek değişkenli AUC):\n" + audit.to_string(index=False))

# Kolinearite: türetilmiş sütunlar
corr = df[feats].corr(method="spearman")
high = [(a, b, corr.loc[a, b]) for i, a in enumerate(feats) for b in feats[i + 1:] if abs(corr.loc[a, b]) >= 0.90]
colin = pd.DataFrame(high, columns=["Feature_A", "Feature_B", "Spearman_rho"]).sort_values("Spearman_rho", key=abs, ascending=False)
L("\nYüksek kolinearite (|rho|>=0.90):\n" + (colin.to_string(index=False) if len(colin) else "yok"))
derived_note = pd.DataFrame({
    "Feature": ["NLR", "PLR", "MLR", "SII", "PIV", "LYM_pct", "MONO_pct"],
    "Derived_from": ["NEU/LYM", "PLT/LYM", "MONO/LYM", "PLT*NEU/LYM", "PLT*NEU*MONO/LYM", "LYM/WBC", "MONO/WBC"],
    "Note": ["deterministic function of raw counts"] * 7,
})

# Sızıntı olarak işaretlenenleri otomatik çıkar
leak_cols = audit.loc[audit["flag"].str.contains("SIZINTI|sıfır varyans"), "Feature"].tolist()
if leak_cols:
    L(f"\n!!! Sızıntı/sabit sütunlar çıkarıldı: {leak_cols}")
    df = df.drop(columns=leak_cols)

# ------------------------------------------------------------------ 6) Kaydet
df.to_csv(OUT / "clean_data.csv", index=False)
L(f"\nTemiz veri kaydedildi: {OUT/'clean_data.csv'}  ({df.shape[0]}×{df.shape[1]})")
L(f"Sınıf dağılımı: {C.NEG_LABEL_NAME}=0: {(df[C.TARGET]==0).sum()}, {C.POS_LABEL_NAME}=1: {(df[C.TARGET]==1).sum()} "
  f"(oran {(df[C.TARGET]==0).sum()/(df[C.TARGET]==1).sum():.2f}:1)")

# Sabit tabakalı bölme — TÜM sonraki adımlar aynı bölmeyi kullanır; test seti asla değiştirilmez
idx_tr, idx_te = train_test_split(df.index.values, test_size=C.TEST_SIZE, stratify=df[C.TARGET],
                                  random_state=C.RANDOM_STATE)
split = pd.DataFrame({"row_id": df.index.values, "split": "train"})
split.loc[split.row_id.isin(idx_te), "split"] = "test"
split.to_csv(OUT / "split_indices.csv", index=False)
L(f"Bölme: train={len(idx_tr)} (TT={df.loc[idx_tr, C.TARGET].sum()}), test={len(idx_te)} (TT={df.loc[idx_te, C.TARGET].sum()})")

with pd.ExcelWriter(OUT / "leakage_audit.xlsx") as xw:
    audit.to_excel(xw, sheet_name="univariate_audit", index=False)
    colin.to_excel(xw, sheet_name="collinearity", index=False)
    derived_note.to_excel(xw, sheet_name="derived_features", index=False)
    corr.round(3).to_excel(xw, sheet_name="spearman_matrix")
(OUT / "data_cleaning_log.txt").write_text("\n".join(log), encoding="utf-8")
print("\nBitti → 00_data/")
