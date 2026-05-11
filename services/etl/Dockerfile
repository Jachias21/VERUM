FROM python:3.11-slim
WORKDIR /app
COPY services/etl/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/etl/app/ ./app/
COPY shared/ ./shared/

CMD ["python", "-m", "app.pipeline"]

