FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY services/worker_vision/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/worker_vision/app/ ./app/
COPY shared/ ./shared/
COPY models/ ./models/

CMD ["python", "-m", "app.worker"]