# 📈 GOOG Stock Predictor

A machine learning system that predicts Google's (GOOG) next-day closing price using 4 different models, a Streamlit dashboard to visualize everything, and an automated script that runs itself every weekday after market close.

Built as a learning project while studying Economics at UBA. Still improving it every week.

---

## What it does

Every weekday at 5:30 PM, the system automatically:
1. Downloads the latest 2 years of GOOG price data
2. Trains 4 ML models on that data
3. Predicts tomorrow's closing price
4. Saves the prediction to a CSV
5. Records today's real closing price (to compare later)

Then you can open the dashboard anytime to see how the models are doing.

---

## The files

| File | What it does |
|------|-------------|
| `StockPredictorV1.ipynb` | First version — exploratory notebook where everything started |
| `stock_predictor_auto.py` | V1 automated script — runs daily, fixed model weights |
| `stock_predictor_auto_v2.py` | V2 automated script — adds sentiment analysis + dynamic weights |
| `goog_screener_dashboard.py` | Streamlit dashboard — Fair Value, BUY/HOLD/SELL signal, Trade Ideas Journal |
| `goog_comparison.py` | Compares V1 vs V2 performance with matplotlib charts |
| `goog_backtest_v2.csv` | Historical predictions from V1 |
| `goog_backtest_v3.csv` | Historical predictions from V2 (with weights and sentiment) |

---

## The models

All 4 models are trained every day on the same features:

- Moving averages (MA5, MA20, MA50)
- RSI (14 days)
- MACD
- Volatility
- Lag features (last 3 closes)
- Daily return
- News sentiment score (V2 only)

| Model | Notes |
|-------|-------|
| Linear Regression | Simple but fast, tends to overfit on big moves |
| Random Forest | Most stable overall |
| SVR | Conservative predictions, stays close to current price |
| XGBoost | Good balance between accuracy and flexibility |

The **Fair Value** is a weighted average of all 4 predictions. In V2, those weights adjust automatically based on which model had the lowest MAE over the last 30 days.

---

## V1 vs V2

| Feature | V1 | V2 |
|---------|----|----|
| Models | 4 ML models | Same 4 models |
| Weights | Fixed (RF 40%, XGB 30%, LR 15%, SVR 15%) | Dynamic — adjusts daily based on MAE |
| News sentiment | No | Yes — via NewsAPI + TextBlob |
| Self-improving | No | Partially — weights adapt over time |

**Honest take:** After 1 week of data, V2 didn't improve much over V1. During high-volatility weeks (like tariff announcements), both models underestimated the real price. The dynamic weights need more historical data to make a real difference — check back in a month.

---

## The dashboard

Run it with:

```bash
streamlit run goog_screener_dashboard.py
```

What you'll see:
- Current price vs Fair Value
- Upside/Downside % and BUY/HOLD/SELL signal
- Breakdown by model with individual predictions
- 60-day price chart vs Fair Value line
- Historical predictions table (once you have data)
- **Trade Ideas Journal** — write your own thesis, set a target and stop loss, track the outcome


---

## Automating it (Windows Task Scheduler)

Both scripts are set up to run automatically every weekday at market close via Windows Task Scheduler:
- `stock_predictor_auto.py` → 5:30 PM
- `stock_predictor_auto_v2.py` → 5:35 PM

---

## What's next

- [ ] Online learning — models that update incrementally instead of retraining from scratch
- [ ] Add macro features (VIX, 10Y Treasury yield, DXY)
- [ ] Expand to multiple tickers
- [ ] Improve sentiment analysis with a better NLP model


*Learning project — not financial advice.*
