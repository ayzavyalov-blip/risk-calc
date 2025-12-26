import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. ЗАГРУЗКА МОДЕЛЕЙ
# ==========================================
# Глобальные переменные для моделей
block_pe = None
block_fgr = None

try:
    print("Загрузка моделей из файла medical_models.pkl...")
    bundle = joblib.load('medical_models.pkl')
    block_pe = bundle['block_pe']
    block_fgr = bundle['block_fgr']
    print("Модели успешно загружены!")
except Exception as e:
    print(f"!!! ОШИБКА ЗАГРУЗКИ МОДЕЛИ: {e}")
    # Не роняем сервер, чтобы видеть логи, но предсказания работать не будут

# ==========================================
# 2. ПОДГОТОВКА ДАННЫХ (Точь-в-точь как при обучении)
# ==========================================
def prepare_features(data):
    # 1. Парсинг простых полей из JSON
    age = float(data.get('age', 30))
    bmi = float(data.get('bmi', 22))
    # В HTML parity приходит как строка, страхуемся
    try:
        parity = int(data.get('parity', 1))
    except:
        parity = 1
    
    # 2. Парсинг выпадающих списков
    interval_str = data.get('interval', '')
    if "Менее 6" in interval_str: interval_val = 0
    elif "Более 60" in interval_str: interval_val = 1
    else: interval_val = 2 # 6-60 мес
    
    pe_hist = 1 if data.get('pe_hist') == 'Да' else 0
    fgr_hist = 1 if data.get('fgr_hist') == 'Да' else 0
    
    hag = 1 if data.get('hag') == 'Да' else 0
    kidney = 1 if data.get('kidney') == 'Да' else 0
    
    gain_str = data.get('gain', '')
    gain_less = 1 if "Меньше" in gain_str else 0
    gain_more = 1 if "Больше" in gain_str else 0
    
    smoke_str = data.get('smoking', '')
    smoke_val = 2 if smoke_str == 'Курит' else 1 

    # 3. Парсинг рисков (формат "1:100")
    def parse_risk(val):
        try:
            val_str = str(val)
            if ':' in val_str:
                parts = val_str.split(':')
                return float(parts[0]) / float(parts[1])
            return float(val_str)
        except:
            return 0.0
            
    risk_fgr_scr = parse_risk(data.get('scr_fgr', 0))

    # 4. Сборка словаря признаков (Имена должны совпадать с names в модели!)
    input_dict = {
        "Возраст матери": age,
        "ИМТ": bmi,
        "ХАГ (0 - нет, 1 - да)": hag,
        "Гест. АГ (0 - нет, 1 - да)": 0, # В I триместре ГАГ обычно 0
        "Хронические  заболевания почек (0 - нет, 1 - да)": kidney,
        "ИсхБер_p_pe_in_history": pe_hist,
        "ИсхБер_n_szrp_in_history": fgr_hist,
        "Интергравидарный интервал (0 - менее 6 мес., 1 - более 60 мес., 2 -6-60 мес)": interval_val,
        "Курение (0 - нет данных, 1 - нет, 2 - да)": smoke_val,
        "Прибавка_меньшеНормы": gain_less,
        "Прибавка_большеНормы": gain_more,
        "Данные 1-го скрининга - риск СЗРП": risk_fgr_scr,
    }
    
    df_input = pd.DataFrame([input_dict])
    
    # 5. Feature Engineering (Создание производных признаков)
    df_input["Возраст35plus"] = (df_input["Возраст матери"] >= 35).astype(int)
    df_input["ИМТ_ожирение"] = (df_input["ИМТ"] >= 30).astype(int)
    df_input["ИМТ_дефицит"] = (df_input["ИМТ"] < 18.5).astype(int)
    df_input["ИМТ_ожирение_ХАГ"] = df_input["ИМТ_ожирение"] * df_input["ХАГ (0 - нет, 1 - да)"]
    
    is_smoking = (df_input["Курение (0 - нет данных, 1 - нет, 2 - да)"] == 2).astype(int)
    df_input["ИМТ_дефицит_курение"] = df_input["ИМТ_дефицит"] * is_smoking

    return df_input

# ==========================================
# 3. API МАРШРУТЫ
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df_full = prepare_features_from_html(data)

        # 1. Расчет риска ПЭ
        features_pe = block_pe["features_sel"]
        X_pe = df_full[features_pe].copy()
        X_pe_imp = block_pe["preprocessor"].transform(X_pe)
        p_pe_raw = block_pe["rf_model"].predict_proba(X_pe_imp)[:, 1]
        p_pe_cal = block_pe["iso_calibrator"].transform(p_pe_raw)[0]

        # 2. Расчет риска ЗРП
        features_fgr = block_fgr["features_sel"]
        X_fgr = df_full[features_fgr].copy()
        X_fgr_imp = block_fgr["preprocessor"].transform(X_fgr)
        p_fgr_raw = block_fgr["rf_model"].predict_proba(X_fgr_imp)[:, 1]
        p_fgr_cal = block_fgr["iso_calibrator"].transform(p_fgr_raw)[0]

        # 3. ДОБАВЛЕНО: Расчет риска ЗРП-НМП (НПФ)
        features_npf = block_npf["features_sel"]
        X_npf = df_full[features_npf].copy()
        X_npf_imp = block_npf["preprocessor"].transform(X_npf)
        p_npf_raw = block_npf["rf_model"].predict_proba(X_npf_imp)[:, 1]
        p_npf_cal = block_npf["iso_calibrator"].transform(p_npf_raw)[0]

        return jsonify({
            'pe_risk': float(p_pe_cal),
            'fgr_risk': float(p_fgr_cal),
            'npf_risk': float(p_npf_cal)  # <-- Добавлено поле для 3-й плашки
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health Check (Яндекс пингует этот адрес, чтобы понять, жив ли контейнер)
@app.route('/health')
def health():
    return 'OK', 200

# Запуск
if __name__ == '__main__':
    # Получаем порт из окружения (Яндекс сам его задаст)
    port = int(os.environ.get('PORT', 8080))

    app.run(host='0.0.0.0', port=port)
