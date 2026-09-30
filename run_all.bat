@echo off
REM TT vs EO — tam pipeline (Windows). Önce: pip install -r requirements.txt
REM Hızlı deneme için satırın başına: set QUICK=1
cd /d %~dp0
set OUT=outputs

echo === 0-1: Temizlik, sızıntı denetimi, tanımlayıcı + tek değişkenli ===
python 00_prepare_data.py            || goto :err
python 01_descriptive_univariate.py  || goto :err

echo === 4: Öznitelik seçimi (PSO-GWO + karşılaştırma; yalnız eğitim seti) ===
python 04_feature_selection.py       || goto :err

echo === 2-3: ANA ANALİZ — PSO-GWO kümesi ile nested CV + XAI ===
python 02_nested_cv_models.py --features-json %OUT%\04_feature_selection\selected_features.json --fs-method PSO-GWO || goto :err
python 03_xai.py --tag _fs_psogwo    || goto :err

echo === 5: Nomogram (PSO-GWO kümesi) ===
python 05_nomogram.py                || goto :err

echo === Duyarlılık analizleri (hakem sorularına hazırlık) ===
python 02_nested_cv_models.py        || goto :err            &REM tüm öznitelikler (seçimsiz)
python 03_xai.py                     || goto :err
python 02_nested_cv_models.py --drop-age   || goto :err      &REM yalnız hematolojik belirteçler
python 03_xai.py --drop-age                || goto :err
python 02_nested_cv_models.py --age-max 35 || goto :err      &REM yaş-örtüşen alt kohort
python 03_xai.py --age-max 35              || goto :err

echo.
echo TAMAMLANDI. Ciktilar: %OUT%
echo Web uygulamasi icin: python 06_webapp.py --model-dir 02_models_fs_psogwo
pause & exit /b 0
:err
echo HATA olustu (yukaridaki mesaja bakin).
pause & exit /b 1
