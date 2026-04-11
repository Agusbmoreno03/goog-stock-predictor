"""
GOOG screener dashboard (Streamlit).

Install and run from a terminal — do not put shell commands in this file as Python code:

    py -m pip install streamlit yfinance scikit-learn xgboost pandas numpy
    streamlit run goog_screener_dashboard.py
"""

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="GOOG Screener", layout="wide")
st.title("Stock screener")
ticker = st.text_input("Ticker", value="GOOG")

if st.button("Load data", type="primary"):
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if df is None or df.empty:
            st.warning("No data returned for that symbol.")
        else:
            st.dataframe(df.tail(30), use_container_width=True)
    except Exception as e:
        st.error(f"Download failed: {e}")
