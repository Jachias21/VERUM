"""
XAI module — generates a Grad-CAM heatmap overlay on the spatial (RGB) branch.
The heatmap is saved to /tmp and its path is returned for Telegram delivery.
"""
from __future__ import annotations

import uuid


async def generate_heatmap(image_bytes: bytes, query_id: uuid.UUID) -> str | None:
    """
    Apply Grad-CAM on the spatial branch of the CNN and overlay it on the
    original image.

    Returns the file path of the saved heatmap PNG, or None if unavailable.

    TODO:
      - Load the PyTorch model (spatial branch only, NOT the ONNX export).
      - Use pytorch_grad_cam.GradCAM targeting the last conv layer.
      - Blend the activation map with the original RGB image.
      - Save to /tmp/heatmap_{query_id}.png and return the path.
    """
    # placeholder — Grad-CAM not yet implemented; return None to skip Telegram photo upload
    return None
