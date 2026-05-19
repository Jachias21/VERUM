FROM python:3.11-slim
WORKDIR /app
COPY services/etl/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/etl/app/ ./app/
COPY shared/ ./shared/
COPY services/etl/cron/ ./cron/

# Install cron daemon (needed by etl_scheduler service)
RUN apt-get update && apt-get install -y --no-install-recommends cron \
    && rm -rf /var/lib/apt/lists/*

# Fix Windows CRLF → LF (Git on Windows may convert line endings)
RUN sed -i 's/\r$//' /app/cron/entrypoint.sh && chmod +x /app/cron/entrypoint.sh

CMD ["python", "-m", "app.pipeline"]

