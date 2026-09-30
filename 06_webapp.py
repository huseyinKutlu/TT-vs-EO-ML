# -*- coding: utf-8 -*-
"""
ADIM 6 — Web tabanlı klinik karar destek uygulaması (Python Dash).

Modül 1 : Nested-CV ile seçilen ML modeli (varsayılan: 02_models_fs_psogwo klasöründeki CatBoost) → anlık P(TT),
          risk kategorisi, yatak başı skoru ve öneri.
Modül 2 : 4 değişkenli etkileşimli nomogram (05_nomogram_reduced) → puanlar, toplam puan, P(TT).

Çalıştırma (tteo ortamında, D:\aliCift klasöründe):
    python 06_webapp.py
    python 06_webapp.py --model xgb          # farklı bir sınıflandırıcı ile Modül 1
    python 06_webapp.py --port 8060
Tarayıcı: http://127.0.0.1:8050

Gerekli klasörler (config.OUT_DIR altında):
    02_models_fs_psogwo/run_info.json, best_model.json, final_models/<clf>.joblib
    05_nomogram_reduced/nomogram_model.json
Hasta verisi saklanmaz; araştırma prototipidir.
"""
import argparse, json, sys
import numpy as np
import pandas as pd
import joblib
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default="02_models_fs_psogwo", help="02 adımı çıktı klasörü")
ap.add_argument("--nomogram-dir", default="05_nomogram_reduced", help="05 adımı çıktı klasörü")
ap.add_argument("--model", default=None, help="final_models içindeki sınıflandırıcı (cat, xgb, et...). Varsayılan: best_model.json")
ap.add_argument("--port", type=int, default=8050)
args = ap.parse_args()

try:
    from dash import Dash, html, dcc, Input, Output
except ImportError:
    sys.exit("Dash kurulu değil:  pip install dash")

# ------------------------------------------------------------------ Dosyaları yükle (açıklayıcı hata mesajlarıyla)
MD = C.OUT_DIR / args.model_dir
ND = C.OUT_DIR / args.nomogram_dir
for f in [MD / "run_info.json", MD / "best_model.json", ND / "nomogram_model.json", C.OUT_DIR / "00_data" / "clean_data.csv"]:
    if not f.exists():
        sys.exit(f"Bulunamadı: {f}\n→ İlgili adımı çalıştırın veya --model-dir / --nomogram-dir yolunu düzeltin.")

info = json.load(open(MD / "run_info.json"))
best = args.model or json.load(open(MD / "best_model.json"))["best_model"]
model_path = MD / "final_models" / f"{best}.joblib"
if not model_path.exists():
    sys.exit(f"Model dosyası yok: {model_path}")
pipe = joblib.load(model_path)
ml_feats = info["features"]
nomo = json.load(open(ND / "nomogram_model.json"))
nm_feats = nomo["features"]
df = pd.read_csv(C.OUT_DIR / "00_data" / "clean_data.csv")

NAMES = {"logreg": "Logistic Regression", "svm": "SVM", "knn": "k-NN", "rf": "Random Forest", "et": "Extra Trees",
         "gb": "Gradient Boosting", "xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost", "mlp": "MLP"}
rng = {f: (float(df[f].quantile(0.005)), float(df[f].quantile(0.995)), float(df[f].median()))
       for f in set(ml_feats) | set(nm_feats)}
label = lambda f: C.PRETTY.get(f, f)
print(f"Modül 1 modeli: {NAMES.get(best, best)} ({len(ml_feats)} öznitelik)  |  Modül 2 nomogram: {len(nm_feats)} öznitelik")

# ------------------------------------------------------------------ Yardımcılar
def risk_level(p):
    if p < 0.10: return "Low", "#2e7d32"
    if p < 0.30: return "Moderate", "#f9a825"
    if p < 0.60: return "High", "#ef6c00"
    return "Very high", "#c62828"

def simple_score(vals):
    s = 0
    for c in nomo["simple_score"]:
        v = vals.get(c["code"])
        if v is None: return None
        s += int(v > c["thr"]) if c["dir"] > 0 else int(v < c["thr"])
    return s

def nomogram_points(x):
    b = np.array(nomo["beta"]); lo = np.array(nomo["lo"]); hi = np.array(nomo["hi"])
    base = np.where(b > 0, lo, hi)
    pts = np.abs(b * (np.clip(x, lo, hi) - base)) * nomo["scale"]
    lp = nomo["intercept"] + float(b @ x)
    return pts, 1 / (1 + np.exp(-lp))

# ------------------------------------------------------------------ Arayüz
CARD = {"background": "#fff", "border": "1px solid #dfe3e8", "borderRadius": "12px", "padding": "18px 22px",
        "margin": "10px", "flex": "1 1 420px", "boxShadow": "0 1px 4px rgba(0,0,0,.06)"}
LABEL = {"fontWeight": 600, "fontSize": "13px", "color": "#333"}
INPUT = {"width": "100%", "padding": "6px 8px", "border": "1px solid #ccc", "borderRadius": "6px"}

def num_input(f, prefix):
    lo, hi, med = rng[f]
    return html.Div([html.Label(label(f), style=LABEL),
                     dcc.Input(id=f"{prefix}-{f}", type="number", value=round(med, 2), step="any", style=INPUT),
                     html.Small(f" reference range in cohort: {lo:.3g} – {hi:.3g}", style={"color": "#888"})],
                    style={"marginBottom": "10px"})

def slider(f):
    lo, hi, med = rng[f]
    return html.Div([html.Label([label(f), ": ", html.Span(id=f"nm-val-{f}", style={"color": "#1565c0"})], style=LABEL),
                     dcc.Slider(id=f"nm-{f}", min=round(lo, 2), max=round(hi, 2), value=round(med, 2),
                                step=round((hi - lo) / 200, 4), marks=None,
                                tooltip={"placement": "bottom", "always_visible": False})],
                    style={"marginBottom": "14px"})

app = Dash(__name__)
app.title = "TT vs EO — Clinical Decision Support"
app.layout = html.Div(style={"fontFamily": "Segoe UI, Arial, sans-serif", "background": "#f4f6f8", "minHeight": "100vh",
                             "padding": "14px 18px"}, children=[
    html.H2("Acute Scrotum: Testicular Torsion vs Epididymo-orchitis — Decision Support",
            style={"margin": "4px 10px", "color": "#1a237e"}),
    html.Div("Research prototype (Adıyaman University). Not a substitute for clinical examination or Doppler ultrasonography. "
             "No patient data are stored.", style={"margin": "0 10px 10px", "color": "#b71c1c", "fontSize": "13px"}),
    html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
        html.Div(style=CARD, children=[
            html.H3(f"Module 1 — Machine-learning model ({NAMES.get(best, best)})", style={"marginTop": 0}),
            html.Div([num_input(f, "ml") for f in ml_feats],
                     style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "columnGap": "16px"}),
            html.Div(id="ml-out", style={"marginTop": "10px"}),
        ]),
        html.Div(style=CARD, children=[
            html.H3("Module 2 — Interactive nomogram (logistic regression)", style={"marginTop": 0}),
            *[slider(f) for f in nm_feats],
            html.Div(id="nm-out", style={"marginTop": "6px"}),
        ]),
    ]),
])

@app.callback(Output("ml-out", "children"), [Input(f"ml-{f}", "value") for f in ml_feats])
def predict_ml(*vals):
    if any(v is None for v in vals):
        return html.Div("Please enter all values.", style={"color": "#888"})
    x = np.array(vals, dtype=float).reshape(1, -1)
    p = float(pipe.predict_proba(x)[0, 1])
    lvl, col = risk_level(p)
    sc = simple_score(dict(zip(ml_feats, vals)))
    advice = ("Urgent urological evaluation / scrotal exploration pathway is suggested."
              if p >= nomo["youden_threshold"] else
              "Torsion less likely; confirm with Doppler ultrasonography and clinical findings.")
    return html.Div([
        html.Div([html.Span("P(testicular torsion) = ", style={"fontSize": "16px"}),
                  html.Span(f"{100*p:.1f}%", style={"fontSize": "26px", "fontWeight": 700, "color": col})]),
        html.Div([html.Span("Risk category: ", style={"fontWeight": 600}), html.Span(lvl, style={"color": col, "fontWeight": 700})]),
        html.Div(f"Bedside score: {sc} / {len(nomo['simple_score'])}" if sc is not None else
                 "Bedside score requires: " + ", ".join(c["code"] for c in nomo["simple_score"])),
        html.Div(advice, style={"marginTop": "8px", "padding": "8px 10px", "background": "#f1f3f4", "borderRadius": "6px"}),
    ])

@app.callback([Output("nm-out", "children")] + [Output(f"nm-val-{f}", "children") for f in nm_feats],
              [Input(f"nm-{f}", "value") for f in nm_feats])
def predict_nomogram(*vals):
    x = np.array(vals, dtype=float)
    pts, p = nomogram_points(x)
    lvl, col = risk_level(p)
    rows = [html.Tr([html.Td(label(f)), html.Td(f"{v:.3g}"), html.Td(f"{pt:.1f}")]) for f, v, pt in zip(nm_feats, vals, pts)]
    table = html.Table([html.Thead(html.Tr([html.Th("Feature"), html.Th("Value"), html.Th("Points")])),
                        html.Tbody(rows + [html.Tr([html.Td(html.B("Total points")), html.Td(""), html.Td(html.B(f"{pts.sum():.1f}"))])])],
                       style={"width": "100%", "borderCollapse": "collapse", "fontSize": "14px"})
    out = html.Div([table,
                    html.Div([html.Span("P(testicular torsion) = ", style={"fontSize": "16px"}),
                              html.Span(f"{100*p:.1f}%", style={"fontSize": "26px", "fontWeight": 700, "color": col}),
                              html.Span(f"  ({lvl})", style={"color": col, "fontWeight": 600})], style={"marginTop": "8px"})])
    return [out] + [f"{v:.3g}" for v in vals]

if __name__ == "__main__":
    print(f"\nUygulama başlatıldı → tarayıcıda açın: http://127.0.0.1:{args.port}   (durdurmak için Ctrl+C)\n")
    app.run(debug=False, host="127.0.0.1", port=args.port)
