"""
conftest.py — test_gateway

Adds `services/gateway` to sys.path so that the module-level absolute imports
inside services/gateway/app/*.py (e.g. `from app.metrics import ...`) resolve
correctly when the test runner's PYTHONPATH is anchored at the project root.

In production the gateway container runs with WORKDIR=services/gateway, so
`app/` is always a top-level package there.  The test container WORKDIR is
the project root (/app), so we patch sys.path here instead.
"""
from __future__ import annotations

import os
import sys

# Insert services/gateway BEFORE the project root so that `from app.xxx`
# finds services/gateway/app/, not some other `app` package.
_GATEWAY_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "services", "gateway")
)
if _GATEWAY_DIR not in sys.path:
    sys.path.insert(0, _GATEWAY_DIR)
