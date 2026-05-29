"""
Pipeline de inferencia dual-stream (rama RGB espacial + rama frecuencial).
La sesión ONNX se carga una vez al importar; si el modelo no existe
el worker devuelve UNVERIFIED sin lanzar excepciones.
"""
from __future__ import annotations

import os
import uuid
import logging

import cv2
import numpy as np

# Importaciones opcionales - el worker arranca aunque falten estas librerias.
try:
    import onnxruntime as ort  # type: ignore[import]
    _ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore[assignment]
    _ORT_AVAILABLE = False

try:
    import torch  # noqa: F401 - requerido por rgb_to_freq_tensor
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# rgb_to_freq_tensor vive en models/vision/preprocess.py.
# El Dockerfile añade /app al PYTHONPATH; en local debe configurarse igual.
try:
    from models.vision.preprocess import rgb_to_freq_tensor  # type: ignore[import] --> cambiar al dockerizar
    _PREPROCESS_AVAILABLE = True
except ImportError:
    rgb_to_freq_tensor = None  # type: ignore[assignment]
    _PREPROCESS_AVAILABLE = False

from shared.schemas import VisionResult  # noqa: E402

log = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IMG_SIZE = 224

# Singleton ONNX - se inicializa una unica vez al importar el modulo.
_SESSION: "ort.InferenceSession | None" = None  # type: ignore[name-defined]
_MODEL_UNAVAILABLE: bool = False


def _load_session() -> None:
    """Carga (o intenta cargar) la sesión ONNX desde VISION_MODEL_PATH."""
    global _SESSION, _MODEL_UNAVAILABLE

    if not _ORT_AVAILABLE:
        log.warning(
            "onnxruntime no instalado - inferencia ONNX deshabilitada."
        )
        _MODEL_UNAVAILABLE = True
        return

    model_path = os.getenv("VISION_MODEL_PATH", "/app/weights/verum_cnn.onnx")

    if not os.path.isfile(model_path):
        log.warning(
            "Modelo ONNX no encontrado en '%s'. "
            "El worker arrancará en modo degradado (devuelve UNVERIFIED).",
            model_path,
        )
        _MODEL_UNAVAILABLE = True
        return

    try:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        _SESSION = ort.InferenceSession(model_path, providers=providers)
        log.info("Sesión ONNX cargada correctamente desde '%s'.", model_path)
    except Exception as exc:  # noqa: BLE001
        log.error("Error al cargar el modelo ONNX: %s", exc)
        _MODEL_UNAVAILABLE = True


# Ejecutar al importar (startup del worker / módulo).
_load_session()


# Preprocesado

def _prepare_rgb_tensor(img_rgb: np.ndarray) -> np.ndarray:
    """
    Convierte img_rgb (H,W,3 uint8) a un array (1,3,224,224) float32
    normalizado con media y desviación ImageNet.
    """
    img = cv2.resize(img_rgb, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, axis=0)


def _prepare_freq_tensor(img_rgb: np.ndarray) -> np.ndarray:
    """
    Genera el tensor frecuencial (1,2,224,224) float32.

    Usa rgb_to_freq_tensor de models/vision/preprocess.py si está disponible;
    en caso contrario aplica el mismo pipeline con numpy puro como fallback.
    """
    if _PREPROCESS_AVAILABLE and _TORCH_AVAILABLE and rgb_to_freq_tensor is not None:
        # rgb_to_freq_tensor devuelve torch.Tensor (2, 224, 224) float32
        freq_np = rgb_to_freq_tensor(img_rgb, size=_IMG_SIZE).numpy()
    else:
        log.debug("Usando pipeline frecuencial de respaldo (sin torch/preprocess).")
        img = cv2.resize(img_rgb, (_IMG_SIZE, _IMG_SIZE))
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        cb = img_ycbcr[:, :, 1].astype(np.float32)
        cr = img_ycbcr[:, :, 2].astype(np.float32)

        def _dft_highpass(ch: np.ndarray) -> np.ndarray:
            fshift = np.fft.fftshift(np.fft.fft2(ch))
            mag = np.log1p(np.abs(fshift))
            h, w = mag.shape
            cy, cx = h // 2, w // 2
            r = min(h, w) // 8
            mag[cy - r : cy + r, cx - r : cx + r] = 0.0
            return mag

        freq_np = np.stack([_dft_highpass(cb), _dft_highpass(cr)], axis=0)

        f_min, f_max = freq_np.min(), freq_np.max()
        if f_max - f_min > 0:
            freq_np = (freq_np - f_min) / (f_max - f_min)

    return np.expand_dims(freq_np.astype(np.float32), axis=0)  # (1, 2, H, W)


async def run_inference(query_id: uuid.UUID, image_bytes: bytes) -> VisionResult:
    """Ejecuta la inferencia dual-branch y devuelve un VisionResult.
    Retorna UNVERIFIED (sin crash) si la imagen está vacía, el modelo no está
    disponible, la imagen no se puede decodificar o la sesión ONNX falla.
    """
    threshold = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.80"))

    if not image_bytes:
        return VisionResult(
            query_id=query_id,
            ai_confidence_score=0.0,
            prnu_detected=False,
            verdict="UNVERIFIED",
        )

    if _MODEL_UNAVAILABLE or _SESSION is None:
        log.warning(
            "query_id=%s - modelo ONNX no disponible; retornando UNVERIFIED.",
            query_id,
        )
        return VisionResult(
            query_id=query_id,
            ai_confidence_score=0.0,
            prnu_detected=False,
            verdict="UNVERIFIED",
        )

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        log.error("query_id=%s - no se pudo decodificar la imagen.", query_id)
        return VisionResult(
            query_id=query_id,
            ai_confidence_score=0.0,
            prnu_detected=False,
            verdict="UNVERIFIED",
        )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    rgb_input  = _prepare_rgb_tensor(img_rgb)
    freq_input = _prepare_freq_tensor(img_rgb)

    # Convención de exportación: el modelo usa "rgb" y "freq" como nombres.
    # Si difieren, se asigna por shape (3 canales → rgb, 2 canales → freq).
    input_names = [inp.name for inp in _SESSION.get_inputs()]
    feed: dict[str, np.ndarray] = {}
    for name in input_names:
        if "freq" in name.lower():
            feed[name] = freq_input
        else:
            feed[name] = rgb_input

    try:
        outputs = _SESSION.run(None, feed)
    except Exception as exc:  # noqa: BLE001
        log.error("query_id=%s - error en sesión ONNX: %s", query_id, exc)
        return VisionResult(
            query_id=query_id,
            ai_confidence_score=0.0,
            prnu_detected=False,
            verdict="UNVERIFIED",
        )

    raw = outputs[0]
    if raw.ndim == 2 and raw.shape[1] == 2:
        score = float(raw[0, 1])
    else:
        score = float(raw.squeeze())

    # Clamp por seguridad ante salidas sin sigmoid/softmax
    score = max(0.0, min(1.0, score))

    verdict = "FAKE" if score >= threshold else "REAL"
    prnu_detected = score < 0.3

    log.info(
        "query_id=%s score=%.4f threshold=%.2f verdict=%s",
        query_id, score, threshold, verdict,
    )

    return VisionResult(
        query_id=query_id,
        ai_confidence_score=round(score, 6),
        prnu_detected=prnu_detected,
        verdict=verdict,
    )
