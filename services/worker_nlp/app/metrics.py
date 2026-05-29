"""
Métricas Prometheus para el worker NLP.

El servidor HTTP (puerto 9101 por defecto, configurable con NLP_METRICS_PORT)
es arrancado por worker.py con prometheus_client.start_http_server().
"""
from prometheus_client import Histogram

nlp_processing_seconds = Histogram(
    "verum_nlp_processing_seconds",
    "End-to-end NLP pipeline latency per message (cache hit or full RAG) in seconds",
    buckets=[0.5, 1, 2, 5, 10, 15, 30, 45, 60],
)
