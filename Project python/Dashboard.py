import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import yfinance as yf

@st.cache_data(ttl=60)
def get_live_price(symbol):
    try:
        live_data = yf.download(symbol, period="1d", interval="1m", auto_adjust=True, progress=False)
        if not live_data.empty:
            return float(live_data['Close'].values[-1])
    except:
        pass
    return None


def load_data(symbol, period):
    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return data

def buy_and_hold(data, capital):
    data = data.copy()
    data['Return'] = data['Close'].pct_change()
    data['Cumulative_Return'] = (1 + data['Return']).cumprod()
    
    buy_price = float(data['Close'].iloc[0])
    shares = capital / buy_price
    final_value = float(shares * data['Close'].iloc[-1])
    
    return {
        "portfolio_values": capital * data['Cumulative_Return'],
        "final_value": final_value,
        "profit": final_value - capital,
        "profit_pct": float((final_value - capital) / capital * 100)
    }


def momentum_strategy(df, capital, short_window=20, long_window=50):
    df = df.copy()
    df = df.dropna()
    df['MA_Short'] = df['Close'].rolling(short_window).mean()
    df['MA_Long'] = df['Close'].rolling(long_window).mean()
    
    df['Signal'] = 0
    df.loc[df['MA_Short'] > df['MA_Long'], 'Signal'] = 1
    
    cash = capital
    shares = 0
    position = 0
    portfolio_values = []

    for i in range(1, len(df)):
        price = float(df['Close'].iloc[i])
        signal = df['Signal'].iloc[i-1]

        if signal == 1 and position == 0:
            shares = cash / price
            cash = 0
            position = 1
        elif signal == 0 and position == 1:
            cash = shares * price
            shares = 0
            position = 0

        portfolio_values.append(shares * price if position == 1 else cash)

    results = pd.DataFrame({
        'Date': df.index[1:],
        'Portfolio_Value': portfolio_values
    })
    results['Strategy_Return'] = results['Portfolio_Value'].pct_change()
    
    return results, float(portfolio_values[-1])


