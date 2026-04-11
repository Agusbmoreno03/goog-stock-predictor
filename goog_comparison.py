"""
goog_comparison.py
==================
Compares V1 (fixed weights) vs V2 (dynamic weights + sentiment).

Output:
    - 3 charts in one figure:
        1. Real price vs Fair Value V1 vs Fair Value V2
        2. MAE per model (V1)
        3. Dynamic weight evolution (V2)
    - Summary table printed to console

How to run:
    python goog_comparison.py

Requirements:
    pip install pandas matplotlib
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(r'C:\Users\agusm\Downloads\Pythonclass\StockPredictor')
CSV_V1   = BASE_DIR / 'goog_backtest_v2.csv'
CSV_V2   = BASE_DIR / 'goog_backtest_v3.csv'
OUTPUT   = BASE_DIR / 'goog_comparison.png'

# Fixed weights used in V1
W_V1 = {'pred_linreg': 0.15, 'pred_rf': 0.40,
         'pred_svr':    0.15, 'pred_xgb': 0.30}
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG  = '#0d0d0d'
CARD_BG  = '#161616'
GRID_CLR = '#1e1e1e'
TEXT_CLR = '#e8e8e8'
MUTED    = '#666666'

C_BLUE   = '#2c7be5'
C_GREEN  = '#00d084'
C_ORANGE = '#ff6b35'
C_AMBER  = '#ffa502'
C_GRAY   = '#888780'


def load_and_prepare():
    v1 = pd.read_csv(CSV_V1, parse_dates=['date'])
    v2 = pd.read_csv(CSV_V2, parse_dates=['date'])

    v1 = v1.dropna(subset=['real_close']).copy()
    v2 = v2.dropna(subset=['real_close']).copy()

    for col in ['real_close', 'pred_linreg', 'pred_rf', 'pred_svr', 'pred_xgb']:
        v1[col] = pd.to_numeric(v1[col], errors='coerce')
        if col in v2.columns:
            v2[col] = pd.to_numeric(v2[col], errors='coerce')

    v1['fair_value_v1'] = sum(v1[c] * w for c, w in W_V1.items())
    v2['fair_value']    = pd.to_numeric(v2['fair_value'], errors='coerce')

    merged = pd.merge(
        v1[['date', 'real_close', 'pred_linreg', 'pred_rf',
            'pred_svr', 'pred_xgb', 'fair_value_v1']],
        v2[['date', 'fair_value', 'sentiment_score',
            'weight_linreg', 'weight_rf', 'weight_svr', 'weight_xgb']],
        on='date', how='inner'
    )
    return v1, v2, merged.sort_values('date').reset_index(drop=True)


def mae(pred, real):
    return np.mean(np.abs(pred - real))


def print_summary(merged):
    print("\n" + "=" * 55)
    print("  GOOG PREDICTOR — V1 vs V2 COMPARISON")
    print("=" * 55)
    print(f"  Period : {merged['date'].min().date()} → {merged['date'].max().date()}")
    print(f"  Days   : {len(merged)}\n")

    print("  MAE PER MODEL (V1 — fixed weights):")
    models = {
        'Linear Regression': 'pred_linreg',
        'Random Forest':     'pred_rf',
        'SVR':               'pred_svr',
        'XGBoost':           'pred_xgb',
    }
    for name, col in models.items():
        print(f"    {name:<22} MAE = ${mae(merged[col], merged['real_close']):.2f}")

    mae_v1   = mae(merged['fair_value_v1'], merged['real_close'])
    mae_v2   = mae(merged['fair_value'],    merged['real_close'])
    delta    = (mae_v1 - mae_v2) / mae_v1 * 100
    winner   = 'V2 predicts better' if delta > 0 else 'V1 predicts better'

    print(f"\n  Fair Value V1 (fixed weights)    MAE = ${mae_v1:.2f}")
    print(f"  Fair Value V2 (dynamic weights)  MAE = ${mae_v2:.2f}")
    print(f"\n  Improvement V2 vs V1  {delta:+.1f}%  ({winner})")
    print("=" * 55 + "\n")


def plot(v1, v2, merged):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), facecolor=DARK_BG)
    fig.suptitle('GOOG — V1 vs V2 Prediction Comparison',
                 fontsize=16, color=TEXT_CLR, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                          left=0.07, right=0.96, top=0.93, bottom=0.07)

    dates_str = [d.strftime('%m/%d') for d in merged['date']]
    x = np.arange(len(dates_str))

    # ── CHART 1: Real price vs Fair Value V1 vs V2 ───────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(CARD_BG)

    ax1.plot(x, merged['real_close'],    color=C_BLUE,  lw=2.5, marker='o',
             markersize=6, label='Real close', zorder=3)
    ax1.plot(x, merged['fair_value_v1'], color=C_GRAY,  lw=1.5, marker='s',
             markersize=5, linestyle='--', label='Fair Value V1 (fixed weights)', zorder=2)
    ax1.plot(x, merged['fair_value'],    color=C_GREEN, lw=2,   marker='^',
             markersize=6, label='Fair Value V2 (dynamic weights)', zorder=2)

    ax1.fill_between(x, merged['real_close'], merged['fair_value'],
                     alpha=0.07, color=C_GREEN)

    for i in range(len(merged)):
        err_v2 = merged['fair_value'].iloc[i] - merged['real_close'].iloc[i]
        ax1.annotate(f'V2: {err_v2:+.1f}',
                     xy=(x[i], merged['fair_value'].iloc[i]),
                     xytext=(0, 8), textcoords='offset points',
                     fontsize=8, color=C_GREEN, ha='center')

    ax1.set_xticks(x)
    ax1.set_xticklabels(dates_str, color=TEXT_CLR)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))
    ax1.tick_params(colors=MUTED)
    ax1.grid(color=GRID_CLR, linewidth=0.5)
    ax1.legend(loc='upper left', fontsize=9, facecolor=CARD_BG,
               labelcolor=TEXT_CLR, framealpha=0.8)
    ax1.set_title('Real Price vs Fair Value V1 vs V2', color=TEXT_CLR,
                  fontsize=11, pad=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRID_CLR)

    # ── CHART 2: MAE per model ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(CARD_BG)

    model_names = ['Linear\nReg', 'Random\nForest', 'SVR', 'XGBoost',
                   'FV V1', 'FV V2']
    model_cols  = ['pred_linreg', 'pred_rf', 'pred_svr', 'pred_xgb',
                   'fair_value_v1', 'fair_value']
    maes   = [mae(merged[c], merged['real_close']) for c in model_cols]
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_AMBER, C_GRAY, C_GREEN]

    bars = ax2.bar(model_names, maes, color=colors, width=0.6,
                   edgecolor=DARK_BG, linewidth=0.5)

    min_idx = int(np.argmin(maes))
    bars[min_idx].set_edgecolor(C_GREEN)
    bars[min_idx].set_linewidth(2)

    for bar, val in zip(bars, maes):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f'${val:.2f}', ha='center', va='bottom',
                 fontsize=9, color=TEXT_CLR)

    ax2.set_title('MAE per Model', color=TEXT_CLR, fontsize=11, pad=8)
    ax2.tick_params(colors=MUTED)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.1f'))
    ax2.grid(axis='y', color=GRID_CLR, linewidth=0.5)
    ax2.set_ylim(0, max(maes) * 1.25)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_CLR)

    # ── CHART 3: Dynamic weight evolution (V2) ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(CARD_BG)

    v2_w = v2.dropna(subset=['weight_linreg']).sort_values('date').reset_index(drop=True)
    xw   = np.arange(len(v2_w))
    dw   = [d.strftime('%m/%d') for d in v2_w['date']]

    ax3.plot(xw, v2_w['weight_linreg'] * 100, color=C_BLUE,   lw=1.5,
             marker='o', markersize=5, label='Linear Reg')
    ax3.plot(xw, v2_w['weight_rf']     * 100, color=C_GREEN,  lw=1.5,
             marker='s', markersize=5, label='Random Forest')
    ax3.plot(xw, v2_w['weight_svr']    * 100, color=C_ORANGE, lw=2,
             marker='^', markersize=6, label='SVR')
    ax3.plot(xw, v2_w['weight_xgb']    * 100, color=C_AMBER,  lw=1.5,
             marker='D', markersize=5, label='XGBoost')

    ax3.set_xticks(xw)
    ax3.set_xticklabels(dw, color=TEXT_CLR, fontsize=9)
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax3.tick_params(colors=MUTED)
    ax3.grid(color=GRID_CLR, linewidth=0.5)
    ax3.legend(loc='upper left', fontsize=8, facecolor=CARD_BG,
               labelcolor=TEXT_CLR, framealpha=0.8)
    ax3.set_title('Dynamic Weight Evolution (V2)', color=TEXT_CLR,
                  fontsize=11, pad=8)
    ax3.set_ylim(0, 55)
    for spine in ax3.spines.values():
        spine.set_edgecolor(GRID_CLR)

    plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f"  Chart saved to: {OUTPUT}")
    plt.show()


def run():
    print("\nLoading data...")
    v1, v2, merged = load_and_prepare()

    if len(merged) == 0:
        print("ERROR: No overlapping dates with complete data between V1 and V2.")
        return

    print_summary(merged)
    plot(v1, v2, merged)


if __name__ == '__main__':
    run()
