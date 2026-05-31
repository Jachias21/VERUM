FROM python:3.11-slim AS base
WORKDIR /app
COPY services/gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/gateway/app/ ./app/
COPY shared/ ./shared/
COPY models/ ./models/

FROM base AS dev
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM base AS prod
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]