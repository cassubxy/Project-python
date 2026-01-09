# Daily Market Report System

## Overview

Automated system that generates daily market reports at 8pm with essential metrics: daily return, volatility, and max drawdown.

## Files

```
Project-python/
├── Project python/
│   ├── Dashboard.py
│   ├── daily_report.py          # Report generator
│   ├── run_daily_report.sh      # Cron wrapper script
│   ├── reports/                 # Generated reports (auto-created)
│   └── logs/                    # Execution logs (auto-created)
├── cron_config.txt              # Cron setup instructions
└── DAILY_REPORTS.md             # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Make Script Executable

```bash
cd "Project python"
chmod +x run_daily_report.sh
```

### 3. Test Manually

```bash
python3 daily_report.py
```

### 4. Setup Cron Job

Edit crontab:

```bash
crontab -e
```

Add this line (replace path):

```
0 20 * * * /path/to/Project-python/Project\ python/run_daily_report.sh
```

Verify:

```bash
crontab -l
```

## Report Content

Each report includes for BTC-USD, ETH-USD, AAPL, MSFT:

- **Close Price**: Closing price of the day
- **Daily Return**: Percentage change from previous day
- **Volatility**: Daily volatility percentage
- **Max Drawdown**: Maximum loss over 5-day period

## Report Format

```
============================================================
DAILY MARKET REPORT
Generated: 2026-01-09 20:00:00
============================================================

BTC-USD
Close Price:      $96,123.75
Daily Return:     +0.93%
Volatility:       2.15%
Max Drawdown:     -3.45%

ETH-USD
Close Price:      $3,456.80
Daily Return:     +1.25%
Volatility:       3.20%
Max Drawdown:     -4.12%
```

## Configuration

### Change Report Time

Modify cron schedule:
- `0 20 * * *` - Daily at 8pm
- `0 9 * * *` - Daily at 9am
- `0 8,20 * * *` - Twice daily (8am & 8pm)

### Change Tracked Assets

Edit `daily_report.py`:

```python
SYMBOLS = ["BTC-USD", "ETH-USD", "AAPL", "MSFT"]
```

## Monitoring

### View Logs

```bash
tail -f logs/cron_$(date +%Y%m%d).log
```

### Check Reports

```bash
ls -lt reports/
cat reports/daily_report_*.txt | tail -20
```

## Troubleshooting

1. **Script not running**
   - Check cron status: `systemctl status cron`
   - Verify permissions: `ls -l run_daily_report.sh`

2. **No data**
   - Test internet: `ping google.com`
   - Check yfinance: `pip show yfinance`
   - Run manually: `python3 daily_report.py`

3. **Permissions**
   - Make executable: `chmod +x run_daily_report.sh`
   - Fix directories: `chmod 755 reports/ logs/`

## Notes

- Reports stored locally (not in Git - see `.gitignore`)
- Requires Python 3 and internet connection
- Uses absolute paths for cron compatibility
- Old reports accumulate - clean manually if needed