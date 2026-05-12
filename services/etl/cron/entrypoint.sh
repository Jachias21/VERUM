#!/bin/sh
set -e

SCHEDULE="${ETL_SCHEDULE:-0 */12 * * *}"

# Redirect cron job output to container stdout/stderr
CRON_LOG="/proc/1/fd/1"

# Register cron job
echo "${SCHEDULE} python -m app.pipeline >> ${CRON_LOG} 2>&1" | crontab -

echo "[etl_scheduler] Cron schedule registered: ${SCHEDULE}"
echo "[etl_scheduler] Running initial ETL pipeline..."

# First run immediately; use || true so a feed failure doesn't stop the container
python -m app.pipeline || true

echo "[etl_scheduler] Initial run complete. Starting cron daemon..."

# Run cron in foreground so the container stays alive
exec cron -f
