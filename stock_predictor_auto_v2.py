"""
stock_predictor_auto_v2.py
==========================
Improved version of the Auto script. Adds:
  1. Automatic weight adjustment — recalculates model weights daily
     based on MAE over the last 30 days.
  2. News sentiment — downloads GOOG headlines via NewsAPI and adds
     a sentiment score as an extra feature.
  3. Macro features — VIX (market fear index) and S&P 500 daily
     return added as extra features to capture broader market context.

Initial setup (once only):
  pip install requests textblob python-dotenv
  python -m textblob.download_corpora

NewsAPI (free, 100 requests/day):
  1. Sign up at https://newsapi.org/register
  2. Create a .env file in the same folder as this script with:
     NEWS_API_KEY=your_key_here
  If no key is found, the script still works — just skips sentiment.

CSV columns:
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
from pathlib import Path


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
TICKER   = 'GOOG'
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / 'goog_backtest_v3.csv'
LOG_PATH = BASE_DIR / 'stock_predictor_auto_v2.log'

NEWS_API_KEY = '84482ec799834f44bed5268f55ae3d88'

DEFAULT_WEIGHTS = {
    'Linear Regression': 0.15,
    'Random Forest':     0.40,
    'SVR':               0.15,
    'XGBoost':           0.30,
}
MAE_WINDOW = 30
MIN_ROWS   = 3

NYSE_HOLIDAYS_2026 = {
    datetime(2026, 1, 1).date(),
    datetime(2026, 1, 19).date(),
    datetime(2026, 2, 16).date(),
    datetime(2026, 4, 3).date(),
    datetime(2026, 5, 25).date(),
    datetime(2026, 6, 19).date(),
    datetime(2026, 7, 3).date(),
    datetime(2026, 9, 7).date(),
    datetime(2026, 11, 26).date(),
    datetime(2026, 12, 25).date(),
}
# ──────────────────────────────────────────────────────────────────────────────


def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def next_trading_day(from_date):
    next_day = from_date + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in NYSE_HOLIDAYS_2026:
        next_day += timedelta(days=1)
    return next_day


# ── MACRO DATA ────────────────────────────────────────────────────────────────
def download_macro(period='2y'):
    """
    Downloads VIX and S&P 500 data aligned by date.
    VIX       → market fear / expected volatility
    SPY_Return → S&P 500 daily return (broad market direction)
    """
    try:
        vix = yf.download('^VIX', period=period, auto_adjust=True, progress=False)
        spy = yf.download('SPY',  period=period, auto_adjust=True, progress=False)

        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)

        vix = vix[['Close']].rename(columns={'Close': 'VIX'})
        spy = spy[['Close']].copy()
        spy['SPY_Return'] = spy['Close'].pct_change(1)
        spy = spy[['SPY_Return']]

        macro = vix.join(spy, how='inner')
        log(f"  Macro: {len(macro)} rows — VIX avg={macro['VIX'].mean():.1f}")
        return macro

    except Exception as e:
        log(f"  Macro: download failed — {e} — using neutral fallback values")
        return None


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def build_features(df, macro=None, sentiment_score=0.0):
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
    d['Sentiment']  = sentiment_score
    d['Next_Close'] = d['Close'].shift(-1)

    # Macro features — join by date index
    if macro is not None:
        d = d.join(macro, how='left')
        d['VIX']        = d['VIX'].ffill()
        d['SPY_Return'] = d['SPY_Return'].fillna(0.0)
    else:
        d['VIX']        = 20.0  # neutral fallback
        d['SPY_Return'] = 0.0

    return d.dropna()


# ── MODELS ────────────────────────────────────────────────────────────────────
def train_and_predict(df_feat):
    FEATURES = ['Close', 'Volume', 'MA5', 'MA20', 'MA50',
                'RSI', 'MACD', 'Volatility', 'Lag1', 'Lag2', 'Lag3',
                'Return1d', 'Sentiment', 'VIX', 'SPY_Return']

    X      = df_feat[FEATURES].values
    y      = df_feat['Next_Close'].values
    ultimo = df_feat[FEATURES].iloc[-1].values.reshape(1, -1)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR':               SVR(kernel='rbf', C=1e3, gamma=0.1),
        'XGBoost':           XGBRegressor(n_estimators=100, learning_rate=0.05,
                                          max_depth=4, random_state=42, verbosity=0),
    }
    NEED_SCALER = {'Linear Regression', 'SVR'}

    preds = {}
    for name, model in models.items():
        if name in NEED_SCALER:
            sc  = StandardScaler()
            Xsc = sc.fit_transform(X)
            model.fit(Xsc, y)
            preds[name] = round(float(model.predict(sc.transform(ultimo))[0]), 2)
        else:
            model.fit(X, y)
            preds[name] = round(float(model.predict(ultimo)[0]), 2)
    return preds


# ── DYNAMIC WEIGHT ADJUSTMENT ─────────────────────────────────────────────────
def compute_dynamic_weights(rows):
    completed = [
        r for r in rows
        if r.get('real_close', '') not in ('', None)
        and r.get('pred_linreg', '') not in ('', None)
    ]

    if len(completed) < MIN_ROWS:
        log(f"  Weights: using defaults ({len(completed)} completed rows, min {MIN_ROWS})")
        return DEFAULT_WEIGHTS, {k: None for k in DEFAULT_WEIGHTS}

    window  = completed[-MAE_WINDOW:]
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
                errors.append(abs(float(r['real_close']) - float(r[col])))
            except (ValueError, KeyError):
                continue
        maes[model_name] = round(np.mean(errors), 4) if errors else 999.0

    inv_maes = {m: 1 / mae if mae > 0 else 0 for m, mae in maes.items()}
    total    = sum(inv_maes.values())
    weights  = {m: round(v / total, 4) for m, v in inv_maes.items()}

    log(f"  MAE last {len(window)} days:")
    for m, mae in maes.items():
        log(f"    {m:<22} MAE=${mae:.2f}  weight={weights[m]:.2%}")
    return weights, maes


# ── NEWS SENTIMENT ────────────────────────────────────────────────────────────
def get_news_sentiment(ticker='GOOG'):
    if not NEWS_API_KEY:
        log("  Sentiment: no API key found in .env — score = 0.0")
        return 0.0
    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={ticker}+stock&language=en"
            f"&sortBy=publishedAt&pageSize=10"
            f"&apiKey={NEWS_API_KEY}"
        )
        data = requests.get(url, timeout=10).json()
        if data.get('status') != 'ok':
            log(f"  Sentiment: NewsAPI error — {data.get('message', '')}")
            return 0.0
        headlines = [a['title'] for a in data.get('articles', []) if a.get('title')]
        if not headlines:
            log("  Sentiment: no headlines found — score = 0.0")
            return 0.0
        scores = [TextBlob(h).sentiment.polarity for h in headlines]
        avg    = round(np.mean(scores), 4)
        log(f"  Sentiment: {len(headlines)} headlines analyzed — score={avg:+.4f}")
        for h in headlines[:3]:
            log(f"    [{TextBlob(h).sentiment.polarity:+.3f}] {h[:80]}")
        return avg
    except Exception as e:
        log(f"  Sentiment: exception — {e} — score = 0.0")
        return 0.0


# ── CSV ───────────────────────────────────────────────────────────────────────
FIELDNAMES = [
    'date', 'real_close',
    'pred_linreg', 'pred_rf', 'pred_svr', 'pred_xgb',
    'weight_linreg', 'weight_rf', 'weight_svr', 'weight_xgb',
    'fair_value', 'sentiment_score',
]


def load_csv():
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, 'r') as f:
        return list(csv.DictReader(f))


def save_csv(rows):
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

    # STEP 1: Download GOOG data
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

    # STEP 2: Update real close — search ALL rows without real_close
    rows    = load_csv()
    updated = False
    for row in rows:
        if row.get('real_close', '') == '' and row['date'] == str(last_market_date):
            row['real_close'] = str(last_close)
            log(f"Updated real close for {last_market_date}: ${last_close}")
            updated = True
    if updated:
        save_csv(rows)
        rows = load_csv()

    # STEP 3: Download macro data
    log("Downloading macro data (VIX + SPY)...")
    macro = download_macro(period='2y')

    # STEP 4: News sentiment
    log("Fetching news sentiment...")
    sentiment = get_news_sentiment(TICKER)

    # STEP 5: Dynamic weights
    log("Computing dynamic model weights...")
    weights, maes = compute_dynamic_weights(rows)

    # STEP 6: Train models
    log("Building features and training models...")
    df_feat     = build_features(df_raw, macro=macro, sentiment_score=sentiment)
    predictions = train_and_predict(df_feat)

    # STEP 7: Calculate weighted Fair Value
    fv = round(sum(predictions[m] * weights[m] for m in predictions), 2)

    log("Predictions:")
    for name, pred in predictions.items():
        change = pred - last_close
        sign   = '+' if change >= 0 else ''
        log(f"  {name:<22} ${pred:.2f}  ({sign}${change:.2f})  weight={weights[name]*100:.1f}%")
    log(f"  {'FAIR VALUE':<22} ${fv:.2f}  (weighted)")

    # STEP 8: Save new prediction row
    pred_date      = next_trading_day(last_market_date)
    existing_dates = [r['date'] for r in rows]

    if str(pred_date) in existing_dates:
        log(f"Prediction for {pred_date} already exists. Skipping.")
    else:
        new_row = {
            'date':            str(pred_date),
            'real_close':      '',
            'pred_linreg':     predictions['Linear Regression'],
            'pred_rf':         predictions['Random Forest'],
            'pred_svr':        predictions['SVR'],
            'pred_xgb':        predictions['XGBoost'],
            'weight_linreg':   round(weights['Linear Regression'], 4),
            'weight_rf':       round(weights['Random Forest'], 4),
            'weight_svr':      round(weights['SVR'], 4),
            'weight_xgb':      round(weights['XGBoost'], 4),
            'fair_value':      fv,
            'sentiment_score': sentiment,
        }
        rows.append(new_row)
        save_csv(rows)
        log(f"Saved prediction for {pred_date} to {CSV_PATH}")

    # STEP 9: Summary
    log("-" * 55)
    log(f"Last close  ({last_market_date}): ${last_close}")
    log(f"Sentiment score: {sentiment:+.4f}")
    log(f"Predictions for {pred_date}:")
    for name, pred in predictions.items():
        log(f"  {name:<22} ${pred:.2f}  (weight {weights[name]:.2%})")
    log(f"  {'FAIR VALUE':<22} ${fv:.2f}")
    log("Script v2 finished successfully.")
    log("-" * 55)


if __name__ == '__main__':
    run()