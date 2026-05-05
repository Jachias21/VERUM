"""
Two-Stream CNN inference pipeline.

Steps:
  1. Convert RGB → YCbCr, isolate Cb/Cr channels.
  2. Apply 2D DFT + high-pass filter → frequency tensor.
  3. Run ONNX model (dual-branch: spatial RGB + frequency).
  4. Return confidence score and verdict.
"""
from __future__ import annotations

import os
import uuid

import cv2
import numpy as np

from shared.schemas import VisionResult


def _preprocess(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb_tensor, freq_tensor) ready for the dual-branch CNN."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Frequency branch ─────────────────────────────────────────────────────
    img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cb, cr = img_ycbcr[:, :, 1], img_ycbcr[:, :, 2]

    def to_freq(channel: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(channel.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        # High-pass: zero out low-frequency centre
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 8
        magnitude[cy - r:cy + r, cx - r:cx + r] = 0
        return magnitude

    freq = np.stack([to_freq(cb), to_freq(cr)], axis=-1)
    return img_rgb, freq


async def run_inference(query_id: uuid.UUID, image_bytes: bytes) -> VisionResult:
    """Load ONNX model and return a VisionResult."""
    # TODO: load model once at startup (module-level singleton)
    model_path = os.getenv("VISION_MODEL_PATH", "/app/weights/verum_cnn.onnx")
    threshold = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", 0.80))

    if not image_bytes:
        return VisionResult(
            query_id=query_id,
            ai_confidence_score=0.0,
            prnu_detected=False,
            verdict="UNVERIFIED",
        )

    rgb, freq = _preprocess(image_bytes)

    # TODO: run onnxruntime session with [rgb, freq] inputs
    score: float = 0.0  # placeholder

    verdict = "FAKE" if score >= threshold else "REAL"
    return VisionResult(
        query_id=query_id,
        ai_confidence_score=score,
        prnu_detected=(score < 0.3),   # low synthetic score → PRNU present
        verdict=verdict,
    )
