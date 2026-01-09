#!/usr/bin/env python3
"""
Daily Market Report Generator
Generates a simple daily report with essential metrics:
- Daily return
- Daily volatility
- Max drawdown

This script runs via cron job daily at 8pm.
Reports are saved in the 'reports/' directory.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configuration
SYMBOLS = ["BTC-USD", "ETH-USD", "AAPL", "MSFT"]
REPORT_DIR = "reports"


def ensure_report_directory():
    """Create reports directory if it doesn't exist"""
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)


def fetch_data(symbol):
    """Fetch market data for a symbol"""
    try:
        data = yf.download(symbol, period="5d", interval="1d", progress=False, auto_adjust=True)
        return data if not data.empty else None
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def calculate_metrics(data):
    """Calculate essential metrics"""
    if data is None or len(data) < 2:
        return None
    
    # Daily return (fixed: use .iloc[0] instead of direct float())
    close_today = float(data['Close'].iloc[-1])
    close_yesterday = float(data['Close'].iloc[-2])
    daily_return = ((close_today - close_yesterday) / close_yesterday) * 100
    
    # Volatility
    returns = data['Close'].pct_change().dropna()
    daily_volatility = float(returns.std()) * 100
    
    # Max drawdown (over the period)
    cumulative = data['Close'] / data['Close'].iloc[0]
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min()) * 100
    
    return {
        'symbol': data.index[-1].strftime('%Y-%m-%d'),
        'close': close_today,
        'daily_return': daily_return,
        'volatility': daily_volatility,
        'max_drawdown': max_drawdown
    }


def generate_report():
    """Generate report for all symbols"""
    ensure_report_directory()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(REPORT_DIR, f'daily_report_{timestamp}.txt')
    
    print(f"\n{'='*60}")
    print(f"Generating Daily Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    with open(filename, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"DAILY MARKET REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for symbol in SYMBOLS:
            print(f"Processing {symbol}...")
            data = fetch_data(symbol)
            
            if data is not None:
                metrics = calculate_metrics(data)
                if metrics:
                    f.write(f"{symbol}\n")
                    f.write(f"Close Price:      ${metrics['close']:,.2f}\n")
                    f.write(f"Daily Return:     {metrics['daily_return']:+.2f}%\n")
                    f.write(f"Volatility:       {metrics['volatility']:.2f}%\n")
                    f.write(f"Max Drawdown:     {metrics['max_drawdown']:.2f}%\n")
                    f.write("\n")
                    print(f"  [OK] {symbol}: {metrics['daily_return']:+.2f}%")
        
        f.write("="*60 + "\n")
    
    print(f"\nReport saved: {filename}\n")


if __name__ == "__main__":
    generate_report()