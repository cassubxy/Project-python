# Multi-Asset Trading and Portfolio Analysis Platform

A **quantitative finance dashboard** in Python using Streamlit, supporting both single-asset strategy comparison and multi-asset portfolio analysis with optional rebalancing. Includes **automated daily market reports**.

---

## Project Objectives

- Retrieve real market data programmatically  
- Implement simple yet realistic trading strategies  
- Compute standard financial performance metrics:
  - Total return  
  - Annualized volatility  
  - Sharpe ratio (risk-free rate = 0)  
  - Maximum drawdown  
- Visualize results interactively  
- Automate recurring financial reporting

---

## Data Source

- Data retrieved from **Yahoo Finance** using the `yfinance` library  
- Daily data (`1d`) for backtesting, intraday data (`1m`) for live updates  
- Prices are **adjusted** for splits and dividends (`auto_adjust=True`)  

---

## Dashboard Architecture

Implemented in `Dashboard.py` with two independent analysis modes, selectable via the sidebar:

1. **Quant A – Single Asset Analysis**  
2. **Quant B – Multi-Asset Portfolio Analysis**  

---

## Quant A – Single Asset Analysis

Compare two strategies on a single asset:

### Asset Selection

- One asset (equity or cryptocurrency)  
- Historical period: 1 month → 2 years  
- Initial capital  
- Live prices refreshed every 5 minutes  

### Strategy 1: Buy & Hold

- Invest full capital at first available price  
- Fixed transaction fee: **0.1% per trade**  
- Portfolio value follows cumulative asset return  

**Outputs:**  
- Portfolio value over time  
- Final portfolio value  
- Total profit & percentage return  
- Transaction fees paid  

### Strategy 2: Momentum (Moving Average Crossover)

- Long when **short MA > long MA**, close when **short MA ≤ long MA**  
- Fully invested or fully in cash (no leverage or short selling)  
- Transaction fee: **0.1% per buy/sell**  
- Track number of trades  

**User Parameters:**  
- Short MA window  
- Long MA window (must be > short MA)  

**Outputs:**  
- Portfolio value over time  
- Final portfolio value  
- Number of trades  
- Transaction fees paid  

### Performance Metrics

- Total return  
- Annualized volatility  
- Sharpe ratio  
- Maximum drawdown  
- Normalized performance comparison (base = 1)

---

## Quant B – Multi-Asset Portfolio Analysis

Analyze a **buy & hold portfolio** with optional rebalancing.

### Asset Selection & Allocation

- Multiple assets  
- Historical period  
- Initial capital  
- Asset weights defined manually via sliders (sum = 100%)  
- Weights stored in Streamlit session state  

### Portfolio Configurations

1. **Buy & Hold (no rebalancing)**  
   - Allocate capital according to weights  
   - Hold until the end of period  

2. **Buy & Hold with Rebalancing**  
   - Rebalance to target weights at fixed intervals:  
     - Monthly (21 trading days)  
     - Quarterly (63 trading days)  
     - Yearly (252 trading days)  
   - Fully liquidate and reallocate capital at each rebalance  
   - No transaction fees applied  

### Analysis Outputs

- Individual asset Buy & Hold results  
- Total portfolio value over time  
- Final portfolio value & total return  
- Annualized portfolio volatility  
- Correlation matrix of asset returns  
- Normalized performance comparison (assets vs portfolio)

---

## Automated Daily Reports

Script: `daily_reports.py`  

- Can be executed manually or via cron/task scheduler  
- Uses last 5 trading days of data  
- Computes:
  - Daily return  
  - Daily volatility  
  - Maximum drawdown  
- Generates one text report per execution in `reports/`  
- Assets covered: BTC-USD, ETH-USD, AAPL, MSFT

---

## Installation & How to Run

### Requirements

- Python 3.9+  
- pip

### Setup

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the interactive dashboard
streamlit run Dashboard.py

# Generate daily market reports
python daily_reports.py

`Tip: Use a cron job or task scheduler to automate daily report generation.`
