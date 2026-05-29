"""
Módulo XAI - genera un mapa de calor Grad-CAM++ sobre la rama espacial (RGB).
Usa GradCAMPlusPlus apuntando al último bloque Conv2dNormActivation de EfficientNet-B0.
El modelo se carga una vez al importar; los errores producen un PNG en blanco.
Selección de dispositivo: CUDA > MPS > CPU.
"""
from __future__ import annotations

import logging
import uuid

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Importaciones opcionales - el worker arranca aunque falten estas librerías.

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


_CHECKPOINT = Path("models/vision/weights/verum_cnn.pt") # ruta relativa al Docker
_IMG_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _select_device() -> "torch.device":  # type: ignore[name-defined]
    if not _TORCH_AVAILABLE:
        return None  # type: ignore[return-value]
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_MODEL: Optional["TwoStreamCNN"] = None  # type: ignore[name-defined]
_DEVICE: Optional["torch.device"] = None  # type: ignore[name-defined]
_MODEL_READY: bool = False


def _load_model() -> None:
    """Carga TwoStreamCNN desde el checkpoint (se llama una vez al importar)."""
    global _MODEL, _DEVICE, _MODEL_READY

    if not _TORCH_AVAILABLE:
        log.warning("xai: torch not available - Grad-CAM disabled.")
        return
    if not _ARCH_AVAILABLE:
        log.warning("xai: TwoStreamCNN architecture not importable - Grad-CAM disabled.")
        return
    if not _GRADCAM_AVAILABLE:
        log.warning("xai: pytorch_grad_cam not installed - Grad-CAM disabled.")
        return

    device = _select_device()
    ckpt_path = _CHECKPOINT

    if not ckpt_path.is_file():
        log.warning(
            "xai: checkpoint not found at '%s' - Grad-CAM will fall back to blank PNG.",
            ckpt_path,
        )
        return

    try:
        model = TwoStreamCNN(pretrained=False)
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        # Soporta tanto state-dicts directos como dicts con clave 'model_state_dict'
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
        log.error("xai: failed to load model for Grad-CAM - %s", exc)


# Cargar al importar el módulo (arranque del worker).
_load_model()


def _decode_rgb(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decodifica bytes → array RGB (H, W, 3) uint8, o None si falla."""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _to_tensor(img_rgb: np.ndarray) -> "torch.Tensor":  # type: ignore[name-defined]
    """
    Convierte (H, W, 3) uint8 RGB → (1, 3, 224, 224) float32 normalizado con ImageNet
    en *CPU* (requerido por pytorch_grad_cam).
    """
    img = cv2.resize(img_rgb, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD   # (224, 224, 3)
    tensor = torch.from_numpy(img.transpose(2, 0, 1))
    return tensor.unsqueeze(0)


def _get_target_layer() -> nn.Module:  # type: ignore[name-defined]
    """Devuelve spatial_branch[0][8] - último Conv2dNormActivation de EfficientNet-B0."""
    return _MODEL.spatial_branch[0][8]  # type: ignore[index]


class _SpatialForwardWrapper(nn.Module):  # type: ignore[name-defined]
    """
    Wrapper para que GradCAMPlusPlus reciba un modelo de entrada única.
    Suministra un tensor de frecuencia cero dummy para que los gradientes
    sobre la rama espacial se calculen correctamente.
    """

    def __init__(self, model: "TwoStreamCNN", device: "torch.device") -> None:  # type: ignore[name-defined]
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, rgb: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        # Tensor de frecuencia cero (mismo batch y dispositivo).
        freq = torch.zeros(rgb.size(0), 2, _IMG_SIZE, _IMG_SIZE, device=self.device)
        return self.model(rgb.to(self.device), freq)


def _write_blank_png(path: Path) -> None:
    """Crea un PNG negro como fallback."""
    blank = np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
    cv2.imwrite(str(path), blank)


async def generate_heatmap(image_bytes: bytes, query_id: uuid.UUID) -> str | None:
    """
    Aplica Grad-CAM++ sobre la rama espacial de TwoStreamCNN y superpone el mapa
    de activación sobre la imagen RGB original.
    Devuelve la ruta al PNG del heatmap, o None si el modelo no está listo.
    Ante cualquier error genera un PNG en blanco para no interrumpir al caller.
    """
    # Grad-CAM pendiente de implementar; devolver None omite el envío de la foto por Telegram
    return None
    output_path = Path(f"/tmp/heatmap_{query_id}.png")

    if not _MODEL_READY or _MODEL is None or _DEVICE is None:
        log.warning(
            "xai: query_id=%s - model not ready; writing blank PNG.", query_id
        )
        _write_blank_png(output_path)
        return str(output_path)

    if not image_bytes:
        log.warning("xai: query_id=%s - empty image_bytes; writing blank PNG.", query_id)
        _write_blank_png(output_path)
        return str(output_path)

    try:
        img_rgb = _decode_rgb(image_bytes)
        if img_rgb is None:
            raise ValueError("cv2.imdecode devolvió None - imagen ilegible.")

        input_tensor = _to_tensor(img_rgb)

        wrapper = _SpatialForwardWrapper(_MODEL, _DEVICE)
        wrapper.eval()
        target_layers = [_get_target_layer()]

        # targets=None → clase de mayor puntuación (binario: clase 1)
        with GradCAMPlusPlus(model=wrapper, target_layers=target_layers) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=None,
            )

        grayscale_cam = grayscale_cam[0]

        img_float = cv2.resize(img_rgb, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32) / 255.0
        visualization = show_cam_on_image(
            img_float,
            grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
            image_weight=0.6,
        )

        # cv2.imwrite espera BGR
        cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        log.info("xai: query_id=%s - heatmap saved to '%s'.", query_id, output_path)

    except Exception as exc:  # noqa: BLE001
        log.error(
            "xai: query_id=%s - Grad-CAM failed (%s); writing blank PNG.", query_id, exc
        )
        _write_blank_png(output_path)

    return str(output_path)
