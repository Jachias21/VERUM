FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY services/worker_nlp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download es_core_news_lg
RUN python -m spacy download en_core_web_sm
COPY services/worker_nlp/app/ ./app/
COPY shared/ ./shared/
COPY models/ ./models/

CMD ["python", "-m", "app.worker"]