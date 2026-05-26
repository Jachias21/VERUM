"""
XAI module — generates a Grad-CAM++ heatmap overlay on the spatial (RGB) branch.

Design decisions
----------------
* Uses ``GradCAMPlusPlus`` from ``pytorch_grad_cam`` (more accurate than vanilla
  Grad-CAM for multi-scale features like EfficientNet).
* Target layer: ``spatial_branch[0][8]`` — the last ``Conv2dNormActivation``
  block (Conv2d → BatchNorm → SiLU) of EfficientNet-B0 inside the
  ``TwoStreamCNN.spatial_branch`` sequential wrapper.
* The model is loaded **once** at module import time (singleton pattern) so
  repeated calls from the async worker do not pay the checkpoint-load cost.
* All errors are handled gracefully — if anything fails a blank PNG is written
  and its path is returned so the caller never crashes.
* Device selection: CUDA → MPS (Apple Silicon) → CPU.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — worker stays alive even if these are missing.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _GRADCAM_AVAILABLE = True
except ImportError:
    GradCAMPlusPlus = None  # type: ignore[assignment]
    show_cam_on_image = None  # type: ignore[assignment]
    _GRADCAM_AVAILABLE = False

try:
    from models.vision.architecture import TwoStreamCNN  # type: ignore[import]
    _ARCH_AVAILABLE = True
except ImportError:
    TwoStreamCNN = None  # type: ignore[assignment]
    _ARCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CHECKPOINT = Path("models/vision/weights/verum_cnn.pt") # --> esto es relativo al Docker
_IMG_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def _select_device() -> "torch.device":  # type: ignore[name-defined]
    if not _TORCH_AVAILABLE:
        return None  # type: ignore[return-value]
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_MODEL: Optional["TwoStreamCNN"] = None  # type: ignore[name-defined]
_DEVICE: Optional["torch.device"] = None  # type: ignore[name-defined]
_MODEL_READY: bool = False


def _load_model() -> None:
    """Load TwoStreamCNN from the best checkpoint (called once at import)."""
    global _MODEL, _DEVICE, _MODEL_READY

    if not _TORCH_AVAILABLE:
        log.warning("xai: torch not available — Grad-CAM disabled.")
        return
    if not _ARCH_AVAILABLE:
        log.warning("xai: TwoStreamCNN architecture not importable — Grad-CAM disabled.")
        return
    if not _GRADCAM_AVAILABLE:
        log.warning("xai: pytorch_grad_cam not installed — Grad-CAM disabled.")
        return

    device = _select_device()
    ckpt_path = _CHECKPOINT

    if not ckpt_path.is_file():
        log.warning(
            "xai: checkpoint not found at '%s' — Grad-CAM will fall back to blank PNG.",
            ckpt_path,
        )
        return

    try:
        model = TwoStreamCNN(pretrained=False)
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        # Support both raw state-dicts and checkpoint dicts with a 'model_state_dict' key
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()

        _MODEL = model
        _DEVICE = device
        _MODEL_READY = True
        log.info("xai: TwoStreamCNN loaded for Grad-CAM on device=%s.", device)
    except Exception as exc:  # noqa: BLE001
        log.error("xai: failed to load model for Grad-CAM — %s", exc)


# Load once at module import (worker startup).
_load_model()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_rgb(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode raw bytes → (H, W, 3) uint8 RGB array, or None on failure."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _to_tensor(img_rgb: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
    """
    Convert (H, W, 3) uint8 RGB → (1, 3, 224, 224) float32 ImageNet-normalised
    torch.Tensor on *CPU* (required by pytorch_grad_cam).
    """
    img = cv2.resize(img_rgb, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD   # (224, 224, 3)
    tensor = torch.from_numpy(img.transpose(2, 0, 1))  # (3, 224, 224)
    return tensor.unsqueeze(0)  # (1, 3, 224, 224)


def _get_target_layer() -> nn.Module:  # type: ignore[name-defined]
    """
    Return the Grad-CAM target layer from spatial_branch.

    TwoStreamCNN.spatial_branch is:
        nn.Sequential(
            0: features (Sequential of EfficientNet blocks 0..8 + AdaptiveAvgPool2d)
            1: AdaptiveAvgPool2d   ← stripped from efficientnet.children()
        )

    The architecture builds it as:
        nn.Sequential(*list(efficientnet.children())[:-1])
    which gives:
        [0] → features  (the EfficientNet feature block, itself a Sequential)
        [1] → AdaptiveAvgPool2d

    The last convolutional block is index 8 of the inner ``features`` Sequential:
        spatial_branch[0][8] → Conv2dNormActivation (Conv2d + BN + SiLU)
    """
    return _MODEL.spatial_branch[0][8]  # type: ignore[index]


class _SpatialForwardWrapper(nn.Module):  # type: ignore[name-defined]
    """
    Thin wrapper so that GradCAMPlusPlus receives a single-input model.

    pytorch_grad_cam always calls ``model(input_tensor)`` with one tensor.
    TwoStreamCNN.forward expects (rgb, freq).  We supply a dummy zero freq
    tensor of the right shape so the gradients on the spatial branch are
    computed correctly.
    """

    def __init__(self, model: "TwoStreamCNN", device: "torch.device") -> None:  # type: ignore[name-defined]
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, rgb: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        # Dummy frequency tensor — zeros, same batch size, on same device.
        freq = torch.zeros(rgb.size(0), 2, _IMG_SIZE, _IMG_SIZE, device=self.device)
        return self.model(rgb.to(self.device), freq)


def _write_blank_png(path: Path) -> None:
    """Create a small black PNG as fallback."""
    blank = np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
    cv2.imwrite(str(path), blank)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_heatmap(image_bytes: bytes, query_id: uuid.UUID) -> str | None:
    """
    Apply Grad-CAM++ on the spatial branch of TwoStreamCNN and overlay the
    activation map on the original RGB image.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the image file (JPEG, PNG, …).
    query_id:
        UUID used to derive the output filename.

    Returns
    -------
    str
        Absolute path of the saved heatmap PNG:  ``/tmp/heatmap_<query_id>.png``.
        A black blank PNG is returned on any error so the caller never crashes.
    """
    output_path = Path(f"/tmp/heatmap_{query_id}.png")

    # ── Guard: dependencies not available ───────────────────────────────────
    if not _MODEL_READY or _MODEL is None or _DEVICE is None:
        log.warning(
            "xai: query_id=%s — model not ready; writing blank PNG.", query_id
        )
        _write_blank_png(output_path)
        return str(output_path)

    # ── Guard: empty payload ─────────────────────────────────────────────────
    if not image_bytes:
        log.warning("xai: query_id=%s — empty image_bytes; writing blank PNG.", query_id)
        _write_blank_png(output_path)
        return str(output_path)

    try:
        # ── 1. Decode image ──────────────────────────────────────────────────
        img_rgb = _decode_rgb(image_bytes)
        if img_rgb is None:
            raise ValueError("cv2.imdecode returned None — unreadable image.")

        # ── 2. Prepare input tensor (CPU — pytorch_grad_cam requirement) ─────
        input_tensor = _to_tensor(img_rgb)  # (1, 3, 224, 224) cpu float32

        # ── 3. Build the wrapper and target layer ────────────────────────────
        wrapper = _SpatialForwardWrapper(_MODEL, _DEVICE)
        wrapper.eval()
        target_layers = [_get_target_layer()]

        # ── 4. Run Grad-CAM++ ────────────────────────────────────────────────
        # use_cuda=False: we handle device placement inside the wrapper.
        with GradCAMPlusPlus(model=wrapper, target_layers=target_layers) as cam:
            # targets=None → use the highest-scoring class (binary: class 1)
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=None,
            )  # shape: (1, H, W) float32 in [0, 1]

        grayscale_cam = grayscale_cam[0]  # (H, W)

        # ── 5. Blend with original RGB ───────────────────────────────────────
        # show_cam_on_image expects float32 RGB in [0, 1] and a (H, W) mask
        img_float = cv2.resize(img_rgb, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32) / 255.0
        visualization = show_cam_on_image(
            img_float,
            grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
            image_weight=0.6,   # 1 - alpha; alpha=0.4 → image_weight=0.6
        )  # (H, W, 3) uint8 RGB

        # ── 6. Save ──────────────────────────────────────────────────────────
        # cv2.imwrite expects BGR
        cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        log.info("xai: query_id=%s — heatmap saved to '%s'.", query_id, output_path)

    except Exception as exc:  # noqa: BLE001
        log.error(
            "xai: query_id=%s — Grad-CAM failed (%s); writing blank PNG.", query_id, exc
        )
        _write_blank_png(output_path)

    return str(output_path)
