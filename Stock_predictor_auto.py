"""
stock_predictor_auto.py
=======================
Runs automatically every weekday after market close.

What it does:
1. Downloads latest GOOG price data
2. Checks if the market was open today
3. Records today's real closing price in the CSV (completes yesterday's row)
4. Trains all 4 models and generates tomorrow's prediction
5. Saves a new row with tomorrow's predictions to the CSV

CSV columns:
    date, real_close, pred_linreg, pred_rf, pred_svr, pred_xgb
"""

import os
import csv
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKER   = 'GOOG'
CSV_PATH = r'C:\Users\%USERNAME%\Downloads\Pythonclass\StockPredictor\goog_backtest_v2.csv' 
CSV_PATH = os.path.expandvars(CSV_PATH)
LOG_PATH = os.path.join(os.path.dirname(CSV_PATH), 'stock_predictor_auto.log')
# ──────────────────────────────────────────────────────────────────────────────


def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def is_market_day(date):
    """Returns True if the given date is a weekday (Mon-Fri)."""
    return date.weekday() < 5


def next_trading_day(from_date):
    """Returns the next weekday after from_date."""
    next_day = from_date + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


def build_features(df):
    d = df.copy()

    # Moving averages
    d['MA5']  = d['Close'].rolling(5).mean()
    d['MA20'] = d['Close'].rolling(20).mean()
    d['MA50'] = d['Close'].rolling(50).mean()

    # RSI (14 days)
    delta = d['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['RSI'] = 100 - (100 / (1 + gain / loss))

    # MACD
    ema12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema26 = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD'] = ema12 - ema26

    # Volatility
    d['Volatility'] = d['Close'].rolling(5).std()

    # Lag features
    d['Lag1'] = d['Close'].shift(1)
    d['Lag2'] = d['Close'].shift(2)
    d['Lag3'] = d['Close'].shift(3)

    # Daily return
    d['Return1d'] = d['Close'].pct_change(1)

    # Target
    d['Next_Close'] = d['Close'].shift(-1)

    return d.dropna()


def train_and_predict(df_feat):
    FEATURES = ['Close', 'Volume', 'MA5', 'MA20', 'MA50',
                'RSI', 'MACD', 'Volatility', 'Lag1', 'Lag2', 'Lag3', 'Return1d']

    X = df_feat[FEATURES].values
    y = df_feat['Next_Close'].values

    ultimo = df_feat[FEATURES].iloc[-1].values.reshape(1, -1)

    modelos = {
        'Linear Regression': LinearRegression(),
        'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR':               SVR(kernel='rbf', C=1e3, gamma=0.1),
        'XGBoost':           XGBRegressor(n_estimators=100, learning_rate=0.05,
                                           max_depth=4, random_state=42, verbosity=0),
    }
    NECESITAN_SCALER = {'Linear Regression', 'SVR'}

    predicciones = {}
    for nombre, modelo in modelos.items():
        if nombre in NECESITAN_SCALER:
            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X)
            modelo.fit(X_sc, y)
            pred = modelo.predict(scaler.transform(ultimo))[0]
        else:
            modelo.fit(X, y)
            pred = modelo.predict(ultimo)[0]
        predicciones[nombre] = round(float(pred), 2)

    return predicciones


def load_csv():
    """Load existing CSV or return empty list."""
    if not os.path.exists(CSV_PATH):
        return []
    rows = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_csv(rows):
    """Save all rows to CSV."""
    fieldnames = ['date', 'real_close',
                  'pred_linreg', 'pred_rf', 'pred_svr', 'pred_xgb']
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run():
    today = datetime.now().date()
    log(f"Script started — today is {today} ({today.strftime('%A')})")

    # ── STEP 1: Download data ─────────────────────────────────────────────────
    log(f"Downloading {TICKER} data (2 years)...")
    df_raw = yf.download(TICKER, period='2y', auto_adjust=True, progress=False)

    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    if df_raw.empty:
        log("ERROR: No data downloaded. Check internet connection.")
        return

    last_market_date = df_raw.index[-1].date()
    last_close       = round(float(df_raw['Close'].iloc[-1]), 2)
    log(f"Latest available close: {last_market_date} — ${last_close}")

    # ── STEP 2: Check if market was open today ────────────────────────────────
    market_open_today = (last_market_date == today)
    if not market_open_today:
        log(f"Market was NOT open today (last close was {last_market_date}). "
            f"Will still generate predictions based on latest data.")

    # ── STEP 3: Update real close for yesterday's prediction ─────────────────
    rows = load_csv()

    if rows:
        last_row = rows[-1]
        # If the last row has no real close yet and matches today's market date
        if last_row['real_close'] == '' and last_row['date'] == str(last_market_date):
            last_row['real_close'] = str(last_close)
            log(f"Updated real close for {last_market_date}: ${last_close}")
            save_csv(rows)

    # ── STEP 4: Build features and train models ───────────────────────────────
    log("Building features and training models...")
    df_feat = build_features(df_raw)
    predicciones = train_and_predict(df_feat)

    log("Predictions:")
    for nombre, pred in predicciones.items():
        cambio = pred - last_close
        signo  = '+' if cambio >= 0 else ''
        log(f"  {nombre:<22} ${pred:.2f}  ({signo}${cambio:.2f})")

    # ── STEP 5: Save new prediction row ──────────────────────────────────────
    pred_date = next_trading_day(last_market_date)

    # Check if we already have a prediction for this date
    existing_dates = [r['date'] for r in rows]
    if str(pred_date) in existing_dates:
        log(f"Prediction for {pred_date} already exists. Skipping.")
    else:
        new_row = {
            'date':        str(pred_date),
            'real_close':  '',
            'pred_linreg': predicciones['Linear Regression'],
            'pred_rf':     predicciones['Random Forest'],
            'pred_svr':    predicciones['SVR'],
            'pred_xgb':    predicciones['XGBoost'],
        }
        rows.append(new_row)
        save_csv(rows)
        log(f"Saved prediction for {pred_date} to {CSV_PATH}")

    # ── STEP 6: Print summary ─────────────────────────────────────────────────
    log("-" * 50)
    log(f"Last close ({last_market_date}): ${last_close}")
    log(f"Predictions for {pred_date}:")
    for nombre, pred in predicciones.items():
        log(f"  {nombre:<22} ${pred:.2f}")
    log("Script finished successfully.")
    log("-" * 50) 


if __name__ == '__main__':
    run()