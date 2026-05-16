"""
Prometheus metrics for the NLP worker.

A lightweight HTTP server (default port 9101, configurable via NLP_METRICS_PORT)
is started by worker.py using prometheus_client.start_http_server() so that
Prometheus can scrape this container independently from the gateway.
"""
from prometheus_client import Histogram

nlp_processing_seconds = Histogram(
    "verum_nlp_processing_seconds",
    "End-to-end NLP pipeline latency per message (cache hit or full RAG) in seconds",
    buckets=[0.5, 1, 2, 5, 10, 15, 30, 45, 60],
)
