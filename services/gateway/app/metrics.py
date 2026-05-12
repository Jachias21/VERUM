"""
Prometheus counters for the VERUM gateway.
Imported by both router.py (increment) and main.py (expose /metrics endpoint).
"""
from prometheus_client import Counter

texts_received = Counter(
    "verum_texts_received_total",
    "Total number of valid text messages published to the NLP queue",
)

images_received = Counter(
    "verum_images_received_total",
    "Total number of image messages published to the vision queue",
)

messages_rejected = Counter(
    "verum_messages_rejected_total",
    "Total number of messages dropped (too short, unsupported type, empty payload)",
)
