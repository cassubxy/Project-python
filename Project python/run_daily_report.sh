#!/bin/bash
# Daily Report Cron Job Script
# Executed by cron every day at 8pm

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/daily_report.py"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/cron_$(date +\%Y\%m\%d).log"

# Create log directory
mkdir -p "$LOG_DIR"

# Log start
echo "========================================" >> "$LOG_FILE"
echo "Daily Report Started" >> "$LOG_FILE"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Run Python script
cd "$SCRIPT_DIR"
python3 "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1

# Log result
if [ $? -eq 0 ]; then
    echo "✓ Report generated successfully" >> "$LOG_FILE"
else
    echo "✗ Report generation failed" >> "$LOG_FILE"
fi

echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"