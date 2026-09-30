# -*- coding: utf-8 -*-
"""
Ortak yapılandırma — TT vs EO makine öğrenmesi projesi.
Tüm yolları ve deney parametrelerini buradan değiştirin.
"""
import os
from pathlib import Path

# ------------------------------------------------------------------ Yollar
DATA_PATH = Path(os.environ.get("DATA_PATH", "data.xlsx"))   # dataset (not distributed)
SHEET_NAME = "Sayfa1"
OUT_DIR = Path(os.environ.get("OUT_DIR", "outputs"))          # tüm çıktılar buraya
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ Hedef
TARGET_COL_RAW = "ORCHİTİS=0 TORSİON=1"
TARGET = "TT"                                   # 1 = testis torsiyonu, 0 = epididimo-orşit
POS_LABEL_NAME, NEG_LABEL_NAME = "Testicular torsion", "Epididymo-orchitis"

# ------------------------------------------------------------------ Sütun adları (ham -> kısa)
RENAME_MAP = {
    "Age": "Age",
    "White blood cell count(WBC)": "WBC",
    "Random distribution of width of red blood cells (RDW)": "RDW",
    "Platelet count": "PLT",
    "PDW": "PDW",
    "PCT": "PCT",
    "Mean platelet volume(MPV)": "MPV",
    "Platelet/Lymphocyte ratio (PLR)": "PLR",
    "LYM%": "LYM_pct",
    "MONO%": "MONO_pct",
    "Neutrophil": "NEU",
    "Lymphocyte": "LYM",
    "Neutrophil/Lymphocyte ratio (NLR)=L1/M1": "NLR",
    "Monocyte": "MONO",
    "Monocyte/Lymphocyte ratio(MLR)": "MLR",
    "PIV.1": "PIV",
    "SII": "SII",
}
DROP_COLS_RAW = ["PIV"]                         # tamamen boş sütun (gerçek PIV = PIV.1)

# Makalede kullanılacak uzun adlar / birimler (tablo ve şekil etiketleri)
PRETTY = {
    "Age": "Age (years)", "WBC": "WBC (10³/µL)", "RDW": "RDW (%)", "PLT": "Platelet count (10³/µL)",
    "PDW": "PDW (fL)", "PCT": "PCT (%)", "MPV": "MPV (fL)", "PLR": "PLR", "LYM_pct": "Lymphocyte (%)",
    "MONO_pct": "Monocyte (%)", "NEU": "Neutrophil (10³/µL)", "LYM": "Lymphocyte (10³/µL)",
    "NLR": "NLR", "MONO": "Monocyte (10³/µL)", "MLR": "MLR", "PIV": "PIV", "SII": "SII",
}

# Öznitelik setleri
FEATURE_SETS = {
    "all":        ["Age","WBC","RDW","PLT","PDW","PCT","MPV","PLR","LYM_pct","MONO_pct","NEU","LYM","NLR","MONO","MLR","PIV","SII"],
    "no_percent": ["Age","WBC","RDW","PLT","PDW","PCT","MPV","PLR","NEU","LYM","NLR","MONO","MLR","PIV","SII"],
    "raw_only":   ["Age","WBC","RDW","PLT","PDW","PCT","MPV","NEU","LYM","MONO"],
    "cbc_only":   ["WBC","RDW","PLT","PDW","PCT","MPV","PLR","NEU","LYM","NLR","MONO","MLR","PIV","SII"],  # Age hariç
}
FEATURE_SET = os.environ.get("FEATURE_SET", "no_percent")

# ------------------------------------------------------------------ Deney parametreleri
RANDOM_STATE = 42
TEST_SIZE = 0.20                 # tabakalı hold-out; test seti hiçbir şekilde değiştirilmez
OUTER_FOLDS = 5
OUTER_REPEATS = 2                # nested CV dış döngü tekrar sayısı
INNER_FOLDS = 5
N_TRIALS = 30                    # Optuna iç döngü deneme sayısı (sınıflandırıcı başına, dış kat başına)
N_BOOT = 2000                    # test seti bootstrap CI
OBJECTIVE_METRIC = "mcc"         # iç döngü optimizasyon ölçütü: "mcc" | "roc_auc" | "average_precision"
N_JOBS = max(1, (os.cpu_count() or 2) - 1)

QUICK = os.environ.get("QUICK", "0") == "1"    # hızlı duman testi
if QUICK:
    OUTER_REPEATS, N_TRIALS, N_BOOT = 1, 4, 200

SAMPLERS = ["none", "ros", "smote", "borderline", "svmsmote", "adasyn", "smoteenn", "smotetomek"]
CLASSIFIERS = ["logreg", "svm", "knn", "rf", "et", "gb", "xgb", "lgbm", "cat", "mlp"]

def out(sub: str) -> Path:
    p = OUT_DIR / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
