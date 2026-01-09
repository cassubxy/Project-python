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

def compute_metrics(portfolio_values):
    series = pd.Series(portfolio_values, dtype="float64").dropna()
    
    returns = series.pct_change().dropna()
    
    cumulative = series / series.iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    sharpe = (
        (returns.mean() * 252) /
        (returns.std() * np.sqrt(252))
        if returns.std() > 0 else 0
    )
    
    return {
        "Total Return": f"{(series.iloc[-1]/series.iloc[0]-1)*100:.2f}%",
        "Volatility": f"{returns.std()*np.sqrt(252)*100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{drawdown.min()*100:.2f}%"
    }

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
#Transaction fees
TRADE_FEE = 0.001  # 0.1 percent per trade

def buy_and_hold(data, capital):
    data = data.copy()
    data['Return'] = data['Close'].pct_change()
    data['Cumulative_Return'] = (1 + data['Return']).cumprod()
    
    buy_price = float(data['Close'].iloc[0])
    shares = (capital * (1 - TRADE_FEE)) / buy_price
    total_fees = capital * TRADE_FEE
    final_value = shares * float(data['Close'].iloc[-1])

    final_value *= (1 - TRADE_FEE)
    total_fees += final_value / (1 - TRADE_FEE) * TRADE_FEE
    
    return {
        "portfolio_values": capital * data['Cumulative_Return'],
        "final_value": final_value,
        "profit": final_value - capital,
        "profit_pct": float((final_value - capital) / capital * 100),
        "total_fees": total_fees
    }


def momentum_strategy(df, capital, short_window=20, long_window=50):
    df = df.copy()
    df = df.dropna()
    df['MA_Short'] = df['Close'].rolling(short_window).mean()
    df['MA_Long'] = df['Close'].rolling(long_window).mean()
    
    df['Signal'] = 0
    df.loc[df['MA_Short'] > df['MA_Long'], 'Signal'] = 1
    
    cash = float(capital)
    shares = 0
    position = 0
    portfolio_values = []
    num_trades = 0
    total_fees = 0.0

    for i in range(1, len(df)):
        price = float(df['Close'].iloc[i])
        signal = df['Signal'].iloc[i-1]

        if signal == 1 and position == 0:
            trade_amount = cash
            fees = trade_amount * TRADE_FEE
            total_fees += fees
            shares = (cash * (1 - TRADE_FEE)) / price 
            cash = 0.0
            position = 1
            num_trades += 1

        elif signal == 0 and position == 1:
            trade_amount = shares * price
            fees = trade_amount * TRADE_FEE
            total_fees += fees
            cash = shares * price * (1 - TRADE_FEE) 
            shares = 0.0
            position = 0
            num_trades += 1

        portfolio_values.append(cash + shares * price)

    results = pd.DataFrame({
        'Date': df.index[1:],
        'Portfolio_Value': portfolio_values
    })
    results['Strategy_Return'] = results['Portfolio_Value'].pct_change()
    
    return results, float(portfolio_values[-1]),num_trades,total_fees

st.set_page_config(
    page_title="Quant A - Single Asset Analysis",
    page_icon="📊",
    layout="wide"
)
st.markdown("""
<meta http-equiv="refresh" content="60">
""", unsafe_allow_html=True)

REFRESH_INTERVAL = 60  # 5 minutes

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

current_time = time.time()
time_since = int(current_time - st.session_state.last_refresh)
next_refresh = max(0, REFRESH_INTERVAL - time_since)

# Afficher l'info
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown(f"**Last update:** {datetime.fromtimestamp(st.session_state.last_refresh).strftime('%H:%M:%S')}")

with col2:
    progress = min(1.0, time_since / REFRESH_INTERVAL)
   

with col3:
    if st.button(" Refresh Now"):
        st.session_state.last_refresh = current_time
        st.cache_data.clear()
        st.rerun()


st.title("Quant A — Single Asset Analysis")
st.markdown("**Comparative Analysis: Buy & Hold vs Momentum**")

st.sidebar.header("Parameters")

symbol = st.sidebar.selectbox(
    "Asset",
    ["BTC-USD", "ETH-USD", "ENGI.PA", "AAPL", "MSFT"],
    index=0
)

period = st.sidebar.selectbox(
    "Period",
    ["1mo", "3mo", "6mo", "1y", "2y"],
    index=3
)

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    value=10000,
    min_value=1000,
    step=1000
)

data = load_data(symbol, period)

if data.empty:
    st.error("Data unavailable")
    st.stop()

current_price = get_live_price(symbol)
if current_price is None:
    current_price = float(data['Close'].iloc[-1])

first_price = float(data['Close'].iloc[0])
price_change_pct = ((current_price - first_price) / first_price) * 100

st.subheader("Asset Information")
col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    if current_price is not None:
        st.metric(f"Current Price ({symbol})", f"${current_price:,.2f}")
    else:
        st.warning("Live price unavailable")

with col_p2:
    st.metric("Period Change", f"{price_change_pct:.2f}%", delta=f"{price_change_pct:.2f}%")

with col_p3:
    st.metric("Period Analyzed", period)

st.subheader("Buy & Hold Strategy")
bh_result = buy_and_hold(data, initial_capital)
final_value_bh = bh_result["final_value"]
portfolio_values_bh = bh_result["portfolio_values"]
profit_pct_bh = bh_result["profit_pct"]
st.metric("Final Value", f"${final_value_bh:,.2f}")
st.write(f"Total transaction fees: ${bh_result['total_fees']:,.2f}")

fig_bh = go.Figure()
fig_bh.add_trace(go.Scatter(
    x=data.index,
    y=portfolio_values_bh,
    name="Portfolio Value",
    line=dict(color="blue")
))
fig_bh.update_layout(height=350)
st.plotly_chart(fig_bh, use_container_width=True)

st.subheader("Momentum Strategy")
col1, col2 = st.columns(2)
with col1:
    short_window = st.slider("Short MA (days)", 5, 50, 20)
with col2:
    long_window = st.slider("Long MA (days)", 21, 200, 50)

# Vérifier que long_window > short_window
if long_window <= short_window:
    st.error("Long MA must be greater than Short MA")
    st.stop()

mom_results, final_value_mom, num_trades, total_fees_mom = momentum_strategy(
    data, initial_capital, short_window, long_window
)

profit_pct_mom = (final_value_mom - initial_capital) / initial_capital * 100


fig_mom = go.Figure()
fig_mom.add_trace(go.Scatter(
    x=mom_results['Date'],
    y=mom_results['Portfolio_Value'],
    name="Momentum Portfolio",
    line=dict(color="purple")
))
fig_mom.update_layout(height=350)
st.plotly_chart(fig_mom, use_container_width=True)
st.write(f"Number of trades executed: {num_trades}")
st.write(f"Total transaction fees: ${total_fees_mom:,.2f}")

st.subheader("Normalized Comparison")
compare_df = pd.DataFrame({
    'Date': mom_results['Date'],
    'Buy & Hold': portfolio_values_bh[1:].values,
    'Momentum': mom_results['Portfolio_Value'].values
})

compare_df['BH_Norm'] = compare_df['Buy & Hold'] / compare_df['Buy & Hold'].iloc[0]
compare_df['Mom_Norm'] = compare_df['Momentum'] / compare_df['Momentum'].iloc[0]

fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(x=compare_df['Date'], y=compare_df['BH_Norm'], name="Buy & Hold"))
fig_compare.add_trace(go.Scatter(x=compare_df['Date'], y=compare_df['Mom_Norm'], name="Momentum"))
st.plotly_chart(fig_compare, use_container_width=True)

metrics_bh = compute_metrics(portfolio_values_bh)
metrics_mom = compute_metrics(mom_results['Portfolio_Value'])

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Buy & Hold Metrics")
    st.metric("Final Value", f"${final_value_bh:,.0f}")
    for k,v in metrics_bh.items():
        st.write(f"*{k}:* {v}")
with col2:
    st.markdown("### Momentum Metrics")
    st.metric("Final Value", f"${final_value_mom:,.0f}")
    st.write(f"Number of trades: {num_trades}")
    for k,v in metrics_mom.items():
        st.write(f"*{k}:* {v}")

st.subheader("Conclusion")
st.info(f"""
- Initial Capital: ${initial_capital:,.0f}
- Buy & Hold: {profit_pct_bh:.2f}%
- Momentum: {profit_pct_mom:.2f}%
- Number of trades (Momentum): {num_trades}

Momentum strategy helps reduce drawdown at the cost of more active management.
""")

st.caption(f"Analysis performed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")







