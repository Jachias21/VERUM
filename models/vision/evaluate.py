"""
Evaluation script for TwoStreamCNN.

Loads verum_cnn_best.pt, runs inference on data/processed/test/,
computes Accuracy, F1, AUC and confusion matrix, and optionally
exports the model to ONNX (opset 17).

Dataset layout expected:
  data/processed/test/
    real/  <images>
    fake/  <images>

Usage:
  # Evaluate only
  python models/vision/evaluate.py

  # Evaluate + export ONNX
  python models/vision/evaluate.py --export-onnx

  # Custom paths
  python models/vision/evaluate.py \\
      --checkpoint weights/verum_cnn_best.pt \\
      --data-root  ../../data/processed \\
      --export-onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets

from architecture import TwoStreamCNN
from preprocess import batch_to_freq_tensors
from train import get_transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Detecta automáticamente CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(checkpoint: Path, device: torch.device) -> TwoStreamCNN:
    """Carga TwoStreamCNN(pretrained=False) desde un checkpoint .pt."""
    model = TwoStreamCNN(pretrained=False).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _print_separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _print_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    """Imprime la matriz de confusión en formato legible."""
    col_w = max(len(n) for n in class_names) + 2
    header = " " * (col_w + 2) + "".join(f"{n:>{col_w}}" for n in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row = f"  {row_name:<{col_w}}" + "".join(f"{v:>{col_w}}" for v in cm[i])
        print(row)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint: Path,
    data_root: Path,
    export_onnx: bool,
    batch_size: int = 32,
    num_workers: int = 4,
) -> None:
    # ── Dispositivo ─────────────────────────────────────────────────────────
    device = get_device()
    print(f"\n[evaluate] Dispositivo: {device}")

    # ── Checkpoint ───────────────────────────────────────────────────────────
    if not checkpoint.is_file():
        print(
            f"[evaluate] ERROR: Checkpoint no encontrado en '{checkpoint}'.\n"
            "  Entrena primero con: python models/vision/train.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[evaluate] Cargando checkpoint: {checkpoint}")
    model = _load_model(checkpoint, device)
    print(f"[evaluate] Modelo cargado correctamente.\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    test_dir = data_root / "test"
    if not test_dir.is_dir():
        print(
            f"[evaluate] ERROR: Directorio de test no encontrado en '{test_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    test_ds = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Determinar qué índice corresponde a "fake"
    # ImageFolder asigna índices en orden alfabético: fake=0, real=1
    class_to_idx = test_ds.class_to_idx
    print(f"[evaluate] Clases detectadas: {class_to_idx}")
    fake_idx = class_to_idx.get("fake", 0)
    print(
        f"[evaluate] Clase positiva (FAKE) = índice {fake_idx} "
        f"| Total muestras: {len(test_ds)}\n"
    )

    # ── Inferencia ───────────────────────────────────────────────────────────
    all_labels: list[int]   = []
    all_preds:  list[int]   = []
    all_scores: list[float] = []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_dl, start=1):
            imgs = imgs.to(device)

            # Rama frecuencial: batch_to_freq_tensors desnormaliza internamente
            freq = batch_to_freq_tensors(imgs).to(device)

            # El modelo devuelve P(fake) ∈ [0, 1] con sigmoid
            scores = model(imgs, freq).squeeze(1).cpu()  # (B,)
            preds  = (scores >= 0.5).int()

            # Si fake_idx = 1, los scores ya representan P(fake).
            # Si fake_idx = 0, invertimos para que AUC sea coherente.
            adjusted_scores = scores if fake_idx == 1 else (1.0 - scores)
            # Labels binarizados: 1 si es fake, 0 si es real
            binary_labels   = (labels == fake_idx).int()

            all_labels.extend(binary_labels.tolist())
            all_preds.extend(preds.tolist())
            all_scores.extend(adjusted_scores.tolist())

            if batch_idx % 10 == 0 or batch_idx == len(test_dl):
                print(f"  Batch {batch_idx:>4d}/{len(test_dl)} procesado.")

    # ── Métricas ─────────────────────────────────────────────────────────────
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    auc  = roc_auc_score(all_labels, all_scores)
    cm   = confusion_matrix(all_labels, all_preds)

    class_names = ["REAL", "FAKE"]  # 0 = real, 1 = fake (binarizado arriba)

    _print_separator("═")
    print("  EVALUACIÓN TwoStreamCNN — VERUM")
    _print_separator("═")
    print(f"  Checkpoint  : {checkpoint}")
    print(f"  Test set    : {test_dir}  ({len(test_ds)} imágenes)")
    print(f"  Dispositivo : {device}")
    _print_separator()
    print(f"  Accuracy    : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  AUC-ROC     : {auc:.4f}")
    _print_separator()
    print("  Matriz de confusión (filas = real, columnas = predicho):")
    print()
    _print_confusion_matrix(cm, class_names)
    print()
    tn, fp, fn, tp = cm.ravel()
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    Precision = {tp/(tp+fp) if (tp+fp)>0 else 0:.4f}")
    print(f"    Recall    = {tp/(tp+fn) if (tp+fn)>0 else 0:.4f}")
    _print_separator("═")

    # ── Exportar a ONNX ──────────────────────────────────────────────────────
    if export_onnx:
        _export_onnx(model, checkpoint, device)


def _export_onnx(model: TwoStreamCNN, checkpoint: Path, device: torch.device) -> None:
    """Exporta el modelo a ONNX con opset 17."""
    # MPS no es compatible con torch.onnx.export; usamos CPU para la exportación.
    export_device = torch.device("cpu") if device.type == "mps" else device
    model_cpu = model.to(export_device)

    dummy_rgb  = torch.randn(1, 3, 224, 224, device=export_device)
    dummy_freq = torch.randn(1, 2, 224, 224, device=export_device)

    onnx_path = checkpoint.with_suffix(".onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[evaluate] Exportando modelo a ONNX → {onnx_path}")
    torch.onnx.export(
        model_cpu,
        (dummy_rgb, dummy_freq),
        str(onnx_path),
        input_names=["rgb", "freq"],
        output_names=["score"],
        dynamic_axes={
            "rgb":   {0: "batch_size"},
            "freq":  {0: "batch_size"},
            "score": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"[evaluate] ✓ ONNX exportado correctamente: {onnx_path}")
    print(
        "\n  NOTA: Copia el fichero .onnx a la ruta apuntada por VISION_MODEL_PATH\n"
        "  para que el worker_vision lo cargue en tiempo de ejecución.\n"
        f"    cp {onnx_path} services/worker_vision/weights/verum_cnn.onnx"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evalúa TwoStreamCNN y (opcionalmente) exporta a ONNX."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("weights/verum_cnn_best.pt"),
        help="Ruta al checkpoint .pt (default: weights/verum_cnn_best.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../../data/processed"),
        help="Raíz del dataset (default: ../../data/processed)",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Exportar el modelo a ONNX tras la evaluación.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de batch para la evaluación (default: 32)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Número de workers para el DataLoader (default: 4)",
    )
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        export_onnx=args.export_onnx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
