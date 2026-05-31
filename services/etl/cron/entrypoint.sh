#!/bin/sh
set -e

SCHEDULE="${ETL_SCHEDULE:-0 */12 * * *}"

# ── Export Docker env vars so cron subprocesses can access them ───────────────
# (cron does not inherit the container's environment by default)
# Python's shlex.quote handles all special characters correctly.
python3 -c "
import os, shlex
with open('/app/.env.cron', 'w') as f:
    for k, v in os.environ.items():
        if k.isidentifier():
            f.write('export {}={}\n'.format(k, shlex.quote(v)))
"
chmod 600 /app/.env.cron

# ── Register cron job ─────────────────────────────────────────────────────────
# Source the env file first, then run the pipeline; send output to container stdout.
echo "${SCHEDULE} . /app/.env.cron; cd /app && python -m app.pipeline >> /proc/1/fd/1 2>&1" | crontab -

echo "[etl_scheduler] Cron schedule registered: ${SCHEDULE}"
echo "[etl_scheduler] Running initial ETL pipeline on startup..."

# First run immediately; use || true so a feed failure doesn't kill the container
python -m app.pipeline || true

echo "[etl_scheduler] Initial run complete. Starting cron daemon..."

# Run cron in foreground so Docker keeps the container alive
exec cron -f
