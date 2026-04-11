"""
stock_predictor_auto_v2.py
==========================
Versión mejorada del Auto. Agrega:
  1. Ajuste de pesos automático — cada día recalcula cuánto peso
     merece cada modelo según su MAE de los últimos 30 días.
  2. Sentiment de noticias — descarga titulares de GOOG via
     NewsAPI y agrega un score de sentimiento como feature extra.

Setup inicial (solo una vez):
  pip install requests textblob
  python -m textblob.download_corpora   ← descarga los datos de lenguaje

NewsAPI (gratis, 100 requests/día):
  1. Registrate en https://newsapi.org/register
  2. Copiá tu API key y pegála en NEWS_API_KEY abajo.
  Si no tenés key, el script igual funciona — solo omite el sentimiento.

CSV columns (nuevas):
    date, real_close,
    pred_linreg, pred_rf, pred_svr, pred_xgb,
    weight_linreg, weight_rf, weight_svr, weight_xgb,
    fair_value, sentiment_score
"""

import os
import csv
import warnings
import requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from textblob import TextBlob

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKER       = 'GOOG'
CSV_PATH = r'C:\Users\%USERNAME%\Downloads\Pythonclass\StockPredictor\goog_backtest_v3.csv' 
CSV_PATH     = os.path.expandvars(CSV_PATH)
LOG_PATH     = os.path.join(os.path.dirname(CSV_PATH), 'stock_predictor_auto_v2.log')
NEWS_API_KEY = '84482ec799834f44bed5268f55ae3d88'

# Pesos base (se usan solo los primeros días, hasta tener historial)
DEFAULT_WEIGHTS = {
    'Linear Regression': 0.15,
    'Random Forest':     0.40,
    'SVR':               0.15,
    'XGBoost':           0.30,
}
MAE_WINDOW = 30   # días de historial para calcular MAE
MIN_ROWS   = 3    # mínimo de filas con real_close para ajustar pesos
# ──────────────────────────────────────────────────────────────────────────────


def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def next_trading_day(from_date):
    next_day = from_date + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def build_features(df, sentiment_score=0.0):
    d = df.copy()
    d['MA5']        = d['Close'].rolling(5).mean()
    d['MA20']       = d['Close'].rolling(20).mean()
    d['MA50']       = d['Close'].rolling(50).mean()

    delta           = d['Close'].diff()
    gain            = delta.where(delta > 0, 0).rolling(14).mean()
    loss            = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['RSI']        = 100 - (100 / (1 + gain / loss))

    ema12           = d['Close'].ewm(span=12, adjust=False).mean()
    ema26           = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD']       = ema12 - ema26
    d['Volatility'] = d['Close'].rolling(5).std()
    d['Lag1']       = d['Close'].shift(1)
    d['Lag2']       = d['Close'].shift(2)
    d['Lag3']       = d['Close'].shift(3)
    d['Return1d']   = d['Close'].pct_change(1)
    d['Sentiment']  = sentiment_score   # ← nuevo feature
    d['Next_Close'] = d['Close'].shift(-1)
    return d.dropna()


# ── MODELOS ───────────────────────────────────────────────────────────────────
def train_and_predict(df_feat):
    FEATURES = ['Close', 'Volume', 'MA5', 'MA20', 'MA50',
                'RSI', 'MACD', 'Volatility', 'Lag1', 'Lag2', 'Lag3',
                'Return1d', 'Sentiment']

    X      = df_feat[FEATURES].values
    y      = df_feat['Next_Close'].values
    ultimo = df_feat[FEATURES].iloc[-1].values.reshape(1, -1)

    modelos = {
        'Linear Regression': LinearRegression(),
        'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR':               SVR(kernel='rbf', C=1e3, gamma=0.1),
        'XGBoost':           XGBRegressor(n_estimators=100, learning_rate=0.05,
                                          max_depth=4, random_state=42, verbosity=0),
    }
    NEED_SCALER = {'Linear Regression', 'SVR'}

    preds = {}
    for name, model in modelos.items():
        if name in NEED_SCALER:
            sc   = StandardScaler()
            Xsc  = sc.fit_transform(X)
            model.fit(Xsc, y)
            preds[name] = round(float(model.predict(sc.transform(ultimo))[0]), 2)
        else:
            model.fit(X, y)
            preds[name] = round(float(model.predict(ultimo)[0]), 2)
    return preds


# ── AJUSTE DE PESOS AUTOMÁTICO ────────────────────────────────────────────────
def compute_dynamic_weights(rows):
    """
    Calcula el peso de cada modelo según su MAE en los últimos MAE_WINDOW días.
    Modelos con menor error → mayor peso.
    Requiere al menos MIN_ROWS filas con real_close completo.
    """
    # Filtrar filas con real_close completo
    completed = [
        r for r in rows
        if r.get('real_close', '') not in ('', None)
        and r.get('pred_linreg', '') not in ('', None)
    ]

    if len(completed) < MIN_ROWS:
        log(f"  Pesos: usando defaults (solo {len(completed)} filas completas, "
            f"mínimo {MIN_ROWS})")
        return DEFAULT_WEIGHTS, {k: None for k in DEFAULT_WEIGHTS}

    # Tomar los últimos MAE_WINDOW días
    window = completed[-MAE_WINDOW:]

    col_map = {
        'Linear Regression': 'pred_linreg',
        'Random Forest':     'pred_rf',
        'SVR':               'pred_svr',
        'XGBoost':           'pred_xgb',
    }

    maes = {}
    for model_name, col in col_map.items():
        errors = []
        for r in window:
            try:
                real = float(r['real_close'])
                pred = float(r[col])
                errors.append(abs(real - pred))
            except (ValueError, KeyError):
                continue
        maes[model_name] = round(np.mean(errors), 4) if errors else 999.0

    # Invertir MAE (menor error → mayor peso) y normalizar
    inv_maes = {m: 1 / mae if mae > 0 else 0 for m, mae in maes.items()}
    total    = sum(inv_maes.values())
    weights  = {m: round(v / total, 4) for m, v in inv_maes.items()}

    log(f"  MAE últimos {len(window)} días:")
    for m, mae in maes.items():
        log(f"    {m:<22} MAE=${mae:.2f}  peso={weights[m]:.2%}")

    return weights, maes


# ── SENTIMENT DE NOTICIAS ─────────────────────────────────────────────────────
def get_news_sentiment(ticker='GOOG'):
    """
    Descarga los últimos 10 titulares de noticias sobre el ticker
    y calcula el sentimiento promedio con TextBlob.
    Retorna un score entre -1 (muy negativo) y +1 (muy positivo).
    """
    if NEWS_API_KEY == 'TU_API_KEY_ACA':
        log("  Sentiment: sin API key configurada — score = 0.0")
        return 0.0

    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={ticker}+stock"
            f"&language=en"
            f"&sortBy=publishedAt"
            f"&pageSize=10"
            f"&apiKey={NEWS_API_KEY}"
        )
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if data.get('status') != 'ok':
            log(f"  Sentiment: error NewsAPI — {data.get('message','')}")
            return 0.0

        articles  = data.get('articles', [])
        headlines = [a['title'] for a in articles if a.get('title')]

        if not headlines:
            log("  Sentiment: sin titulares encontrados — score = 0.0")
            return 0.0

        scores = [TextBlob(h).sentiment.polarity for h in headlines]
        avg    = round(np.mean(scores), 4)

        log(f"  Sentiment: {len(headlines)} titulares analizados — score={avg:+.4f}")
        for h in headlines[:3]:   # mostrar los 3 primeros en el log
            pol = round(TextBlob(h).sentiment.polarity, 3)
            log(f"    [{pol:+.3f}] {h[:80]}")

        return avg

    except Exception as e:
        log(f"  Sentiment: excepción — {e} — score = 0.0")
        return 0.0


# ── CSV ───────────────────────────────────────────────────────────────────────
FIELDNAMES = [
    'date', 'real_close',
    'pred_linreg', 'pred_rf', 'pred_svr', 'pred_xgb',
    'weight_linreg', 'weight_rf', 'weight_svr', 'weight_xgb',
    'fair_value', 'sentiment_score',
]


def load_csv():
    if not os.path.exists(CSV_PATH):
        return []
    rows = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_csv(rows):
    # Asegurar que todas las filas viejas tengan las columnas nuevas
    for r in rows:
        for fn in FIELDNAMES:
            if fn not in r:
                r[fn] = ''
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run():
    today = datetime.now().date()
    log(f"Script v2 started — today is {today} ({today.strftime('%A')})")

    # STEP 1: Descargar datos
    log(f"Downloading {TICKER} data (2 years)...")
    df_raw = yf.download(TICKER, period='2y', auto_adjust=True, progress=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    if df_raw.empty:
        log("ERROR: No data downloaded.")
        return

    last_market_date = df_raw.index[-1].date()
    last_close       = round(float(df_raw['Close'].iloc[-1]), 2)
    log(f"Latest available close: {last_market_date} — ${last_close}")

    # STEP 2: Completar precio real de ayer
    rows = load_csv()
    if rows:
        last_row = rows[-1]
        if last_row.get('real_close', '') == '' and last_row['date'] == str(last_market_date):
            last_row['real_close'] = str(last_close)
            log(f"Updated real close for {last_market_date}: ${last_close}")
            save_csv(rows)
            rows = load_csv()   # recargar con datos frescos

    # STEP 3: Sentiment de noticias
    log("Fetching news sentiment...")
    sentiment = get_news_sentiment(TICKER)

    # STEP 4: Calcular pesos dinámicos
    log("Computing dynamic model weights...")
    weights, maes = compute_dynamic_weights(rows)

    # STEP 5: Entrenar modelos
    log("Building features and training models...")
    df_feat      = build_features(df_raw, sentiment_score=sentiment)
    predicciones = train_and_predict(df_feat)

    # STEP 6: Calcular Fair Value ponderado
    fv = round(sum(predicciones[m] * weights[m] for m in predicciones), 2)

    log("Predictions:")
    for nombre, pred in predicciones.items():
        cambio = pred - last_close
        signo  = '+' if cambio >= 0 else ''
        w_pct  = weights[nombre] * 100
        log(f"  {nombre:<22} ${pred:.2f}  ({signo}${cambio:.2f})  peso={w_pct:.1f}%")
    log(f"  {'FAIR VALUE':<22} ${fv:.2f}  (ponderado)")

    # STEP 7: Guardar nueva fila
    pred_date      = next_trading_day(last_market_date)
    existing_dates = [r['date'] for r in rows]

    if str(pred_date) in existing_dates:
        log(f"Prediction for {pred_date} already exists. Skipping.")
    else:
        new_row = {
            'date':           str(pred_date),
            'real_close':     '',
            'pred_linreg':    predicciones['Linear Regression'],
            'pred_rf':        predicciones['Random Forest'],
            'pred_svr':       predicciones['SVR'],
            'pred_xgb':       predicciones['XGBoost'],
            'weight_linreg':  round(weights['Linear Regression'], 4),
            'weight_rf':      round(weights['Random Forest'], 4),
            'weight_svr':     round(weights['SVR'], 4),
            'weight_xgb':     round(weights['XGBoost'], 4),
            'fair_value':     fv,
            'sentiment_score': sentiment,
        }
        rows.append(new_row)
        save_csv(rows)
        log(f"Saved prediction for {pred_date} to {CSV_PATH}")

    # STEP 8: Resumen
    log("-" * 55)
    log(f"Last close  ({last_market_date}): ${last_close}")
    log(f"Sentiment score: {sentiment:+.4f}")
    log(f"Predictions for {pred_date}:")
    for nombre, pred in predicciones.items():
        log(f"  {nombre:<22} ${pred:.2f}  (peso {weights[nombre]:.2%})")
    log(f"  {'FAIR VALUE':<22} ${fv:.2f}")
    log("Script v2 finished successfully.")
    log("-" * 55)


if __name__ == '__main__':
    run()
