"""
Preprocesado de la rama frecuencial - Two-Stream CNN.

Convierte una imagen RGB en un tensor de frecuencias (2, H, W)
extrayendo los canales de crominancia Cb y Cr en el espacio YCbCr,
aplicando DFT 2D + filtro paso alto para aislar artefactos sintéticos.
"""
import cv2
import numpy as np
import torch


def rgb_to_freq_tensor(img_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    Entrada : imagen RGB (H, W, 3) en uint8
    Salida  : tensor (2, size, size) float32 normalizado - rama frecuencial
    """
    # 1. Redimensionar
    img = cv2.resize(img_rgb, (size, size))

    # 2. RGB → YCbCr, extraer canales de crominancia
    img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    cb = img_ycbcr[:, :, 1].astype(np.float32)
    cr = img_ycbcr[:, :, 2].astype(np.float32)

    # 3. DFT 2D + filtro paso alto sobre cada canal
    freq_cb = _apply_dft_highpass(cb)
    freq_cr = _apply_dft_highpass(cr)

    # 4. Stack → tensor (2, H, W)
    freq = np.stack([freq_cb, freq_cr], axis=0)
    tensor = torch.from_numpy(freq).float()

    # 5. Normalizar al rango [0, 1]
    t_min, t_max = tensor.min(), tensor.max()
    if t_max - t_min > 0:
        tensor = (tensor - t_min) / (t_max - t_min)

    return tensor


def _apply_dft_highpass(channel: np.ndarray) -> np.ndarray:
    """
    Aplica DFT 2D + filtro paso alto sobre un canal 2D.
    El filtro elimina el centro del espectro (bajas frecuencias = contenido
    semántico visible) dejando solo las altas frecuencias donde
    se esconden los artefactos de upsampling de los modelos generativos.
    """
    # DFT + centrar espectro
    f      = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)

    # Magnitud en escala logarítmica
    magnitude = np.log1p(np.abs(fshift))

    # Filtro paso alto: anular cuadrado central
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8          # radio del filtro - cubre ~12% del espectro
    magnitude[cy - r:cy + r, cx - r:cx + r] = 0.0

    return magnitude


def batch_to_freq_tensors(imgs_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convierte un batch de imágenes normalizadas (B, 3, H, W) en tensores
    de frecuencia (B, 2, H, W) para alimentar la rama frecuencial.

    Se usa dentro del bucle de entrenamiento en train.py.
    """
    batch_size = imgs_tensor.shape[0]
    size       = imgs_tensor.shape[2]
    freq_batch = torch.zeros(batch_size, 2, size, size)

    # Desnormalizar ImageNet para recuperar valores de píxel reales
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(batch_size):
        img_t = imgs_tensor[i].cpu() * std + mean          # (3, H, W) en [0,1]
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        freq_batch[i] = rgb_to_freq_tensor(img_np, size)

    return freq_batch