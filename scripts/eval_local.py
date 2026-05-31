#!/usr/bin/env python3
"""
eval_local.py — Evaluación local directa del modelo de visión VERUM.

Evalúa imágenes reales/fake sin pasar por Docker ni Telegram.
Soporta modelos .pt (TwoStreamCNN) y .onnx.

USO:
  python scripts/eval_local.py --model-type pt  --test-dir data/test_custom
  python scripts/eval_local.py --model-type onnx --checkpoint models/vision/weights/verum_cnn_best.onnx

ESTRUCTURA del directorio de test:
  test_dir/
    real/   ← imágenes reales
    fake/   ← imágenes sintéticas/generadas

SALIDAS (en --output-dir):
  results.csv          — resultados imagen por imagen
  score_histogram.png  — distribución de scores reales vs fakes
  confusion_matrix.png — matriz de confusión

MLflow:
  Experimento : VERUM-LocalEval
  Parámetros  : checkpoint, model_type, test_dir, threshold
  Métricas    : accuracy, f1, auc_roc, precision, recall
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ── Ensure project root is importable ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("verum.eval_local")

# ── Constants ─────────────────────────────────────────────────────────────────
_IMG_SIZE = 224
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# ── Label convention: 1 = fake/synthetic, 0 = real ───────────────────────────
_LABEL_FAKE = 1
_LABEL_REAL = 0


# =============================================================================
# Device detection
# =============================================================================

def _select_device() -> "torch.device":  # type: ignore[name-defined]
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    log.info("Dispositivo seleccionado: %s", dev)
    return dev


# =============================================================================
# Image loading & preprocessing
# =============================================================================

def _load_rgb(path: Path):
    """Load image as (H, W, 3) uint8 RGB numpy array."""
    import cv2
    import numpy as np
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"cv2.imread devolvió None para '{path}'")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _to_spatial_tensor(img_rgb):
    """(H,W,3) uint8 → (1,3,224,224) float32 ImageNet-normalised."""
    import cv2
    import numpy as np
    import torch

    img = cv2.resize(img_rgb, (_IMG_SIZE, _IMG_SIZE)).astype(np.float32) / 255.0
    mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(_IMAGENET_STD,  dtype=np.float32)
    img = (img - mean) / std
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,H,W)
    return tensor


def _to_freq_tensor(img_rgb):
    """(H,W,3) uint8 → (1,2,224,224) float32 (frequency branch)."""
    # Reuse the project's preprocess module
    from models.vision.preprocess import rgb_to_freq_tensor
    return rgb_to_freq_tensor(img_rgb, size=_IMG_SIZE).unsqueeze(0)  # (1,2,H,W)


# =============================================================================
# Model loaders
# =============================================================================

def _load_pt_model(checkpoint: Path, device):
    """Load TwoStreamCNN from a .pt checkpoint."""
    import torch
    from models.vision.architecture import TwoStreamCNN

    model = TwoStreamCNN(pretrained=False)
    state = torch.load(str(checkpoint), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    log.info("Modelo PT cargado desde '%s' en %s", checkpoint, device)
    return model


def _load_onnx_session(checkpoint: Path):
    """Load an ONNX inference session (CPU-based via onnxruntime)."""
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(checkpoint), providers=providers)
    except Exception:
        sess = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
    log.info("Sesión ONNX cargada desde '%s'", checkpoint)
    return sess


# =============================================================================
# Inference
# =============================================================================

def _infer_pt(model, img_rgb, device) -> float:
    """Run TwoStreamCNN inference; returns P(fake) in [0, 1]."""
    import torch

    rgb_t  = _to_spatial_tensor(img_rgb).to(device)
    freq_t = _to_freq_tensor(img_rgb).to(device)

    with torch.no_grad():
        score = model(rgb_t, freq_t).item()
    return float(score)


def _infer_onnx(session, img_rgb) -> float:
    """Run ONNX inference; returns P(fake) in [0, 1].

    Expects the ONNX model to have two inputs: 'rgb' and 'freq',
    matching TwoStreamCNN's forward signature.
    Falls back to single-input if the model only has one input.
    """
    import numpy as np

    rgb_np  = _to_spatial_tensor(img_rgb).numpy()   # (1,3,224,224)
    freq_np = _to_freq_tensor(img_rgb).numpy()       # (1,2,224,224)

    input_names = [inp.name for inp in session.get_inputs()]

    if len(input_names) == 2:
        feeds = {input_names[0]: rgb_np, input_names[1]: freq_np}
    else:
        # Single-input ONNX (rgb only)
        feeds = {input_names[0]: rgb_np}

    output = session.run(None, feeds)
    # Output shape: (1,1) or (1,) — take first scalar
    score = float(output[0].flatten()[0])
    return score


# =============================================================================
# Dataset collection
# =============================================================================

def _collect_images(test_dir: Path) -> list[tuple[Path, int]]:
    """Return list of (image_path, label) pairs from test_dir/real/ and test_dir/fake/."""
    real_dir = test_dir / "real"
    fake_dir = test_dir / "fake"

    for d in (real_dir, fake_dir):
        if not d.is_dir():
            raise FileNotFoundError(
                f"Directorio requerido no encontrado: '{d}'\n"
                f"La estructura esperada es:\n"
                f"  {test_dir}/real/\n"
                f"  {test_dir}/fake/"
            )

    samples: list[tuple[Path, int]] = []
    for img_path in sorted(real_dir.rglob("*")):
        if img_path.suffix.lower() in _SUPPORTED_EXTS:
            samples.append((img_path, _LABEL_REAL))
    for img_path in sorted(fake_dir.rglob("*")):
        if img_path.suffix.lower() in _SUPPORTED_EXTS:
            samples.append((img_path, _LABEL_FAKE))

    log.info(
        "Dataset: %d imágenes reales, %d imágenes fake",
        sum(1 for _, l in samples if l == _LABEL_REAL),
        sum(1 for _, l in samples if l == _LABEL_FAKE),
    )
    return samples


# =============================================================================
# Evaluation loop
# =============================================================================

def _evaluate(
    samples: list[tuple[Path, int]],
    model_type: str,
    model_or_session: Any,
    device: Any,
    threshold: float,
) -> list[dict[str, Any]]:
    """Evaluate all samples; return per-image result records."""
    records: list[dict[str, Any]] = []
    total = len(samples)

    for i, (img_path, label) in enumerate(samples, 1):
        label_str = "fake" if label == _LABEL_FAKE else "real"
        print(f"  [{i:4d}/{total}] {img_path.name:<50}", end="", flush=True)

        t0 = time.monotonic()
        try:
            img_rgb = _load_rgb(img_path)

            if model_type == "pt":
                score = _infer_pt(model_or_session, img_rgb, device)
            else:
                score = _infer_onnx(model_or_session, img_rgb)

            predicted_label = _LABEL_FAKE if score >= threshold else _LABEL_REAL
            predicted_str   = "fake" if predicted_label == _LABEL_FAKE else "real"
            correct         = predicted_label == label
            elapsed_ms      = int((time.monotonic() - t0) * 1000)
            error           = None

        except Exception as exc:  # noqa: BLE001
            score           = -1.0
            predicted_label = -1
            predicted_str   = "ERROR"
            correct         = False
            elapsed_ms      = int((time.monotonic() - t0) * 1000)
            error           = str(exc)
            log.warning("Error en '%s': %s", img_path.name, exc)

        mark = "✓" if correct else "✗"
        print(f" score={score:.4f}  pred={predicted_str:<4}  label={label_str:<4}  {mark}  ({elapsed_ms}ms)")

        records.append({
            "filename":        img_path.name,
            "filepath":        str(img_path),
            "score":           round(score, 6),
            "predicted":       predicted_str,
            "label":           label_str,
            "label_int":       label,
            "predicted_int":   predicted_label,
            "correct":         correct,
            "elapsed_ms":      elapsed_ms,
            "error":           error or "",
        })

    return records


# =============================================================================
# Metrics
# =============================================================================

def _compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Accuracy, F1, AUC-ROC, Precision, Recall, confusion matrix."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        confusion_matrix,
    )

    valid = [r for r in records if r["predicted_int"] != -1]
    if not valid:
        raise ValueError("No hay predicciones válidas para calcular métricas.")

    y_true   = [r["label_int"]     for r in valid]
    y_pred   = [r["predicted_int"] for r in valid]
    y_scores = [r["score"]         for r in valid]

    accuracy  = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc_roc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        auc_roc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "total":     len(records),
        "valid":     len(valid),
        "errors":    len(records) - len(valid),
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "auc_roc":   round(auc_roc,   4) if not (auc_roc != auc_roc) else "nan",
        "confusion_matrix": cm,   # [[TN, FP], [FN, TP]]
    }


# =============================================================================
# Plots
# =============================================================================

def _save_histogram(records: list[dict[str, Any]], output_dir: Path) -> Path:
    """Save a score distribution histogram (real vs fake) as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    real_scores = [r["score"] for r in records if r["label"] == "real" and r["score"] >= 0]
    fake_scores = [r["score"] for r in records if r["label"] == "fake" and r["score"] >= 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 41)

    ax.hist(real_scores, bins=bins, alpha=0.65, color="#2ecc71", label=f"Real  (n={len(real_scores)})", edgecolor="white")
    ax.hist(fake_scores, bins=bins, alpha=0.65, color="#e74c3c", label=f"Fake  (n={len(fake_scores)})", edgecolor="white")

    ax.axvline(0.5, color="#333", linestyle="--", linewidth=1.2, label="Umbral = 0.5")
    ax.set_xlabel("Score P(fake)", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.set_title("Distribución de scores — VERUM Vision", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / "score_histogram.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info("Histograma guardado: %s", out_path)
    return out_path


def _save_confusion_matrix(metrics: dict[str, Any], output_dir: Path) -> Path:
    """Save confusion matrix as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay

    cm_array = np.array(metrics["confusion_matrix"])
    labels   = ["Real (0)", "Fake (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Matriz de confusión — VERUM Vision", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = output_dir / "confusion_matrix.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info("Matriz de confusión guardada: %s", out_path)
    return out_path


# =============================================================================
# CSV export
# =============================================================================

def _export_csv(records: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write per-image results to CSV."""
    out_path = output_dir / "results.csv"
    fieldnames = ["filename", "filepath", "score", "predicted", "label", "correct", "elapsed_ms", "error"]

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    log.info("CSV exportado: %s", out_path)
    return out_path


# =============================================================================
# MLflow logging
# =============================================================================

def _log_mlflow(
    metrics: dict[str, Any],
    args: argparse.Namespace,
    histogram_path: Path,
    cm_path: Path,
    csv_path: Path,
) -> None:
    """Log params, metrics and artifacts to MLflow under 'VERUM-LocalEval'."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow no disponible — saltando registro.")
        return

    mlflow.set_experiment("VERUM-LocalEval")

    with mlflow.start_run(run_name=Path(args.checkpoint).stem):
        # Parameters
        mlflow.log_param("checkpoint",  str(args.checkpoint))
        mlflow.log_param("model_type",  args.model_type)
        mlflow.log_param("test_dir",    str(args.test_dir))
        mlflow.log_param("threshold",   args.threshold)
        mlflow.log_param("total_imgs",  metrics["total"])
        mlflow.log_param("valid_imgs",  metrics["valid"])
        mlflow.log_param("error_imgs",  metrics["errors"])

        # Metrics
        mlflow.log_metric("accuracy",  metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall",    metrics["recall"])
        mlflow.log_metric("f1",        metrics["f1"])
        if metrics["auc_roc"] != "nan":
            mlflow.log_metric("auc_roc", float(metrics["auc_roc"]))

        # Confusion matrix values
        cm = metrics["confusion_matrix"]
        mlflow.log_metric("tn", cm[0][0])
        mlflow.log_metric("fp", cm[0][1])
        mlflow.log_metric("fn", cm[1][0])
        mlflow.log_metric("tp", cm[1][1])

        # Artifacts
        mlflow.log_artifact(str(histogram_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(csv_path))

    log.info("Resultados registrados en MLflow (experimento: VERUM-LocalEval).")


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación local directa del modelo de visión VERUM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="models/vision/weights/verum_cnn_best.pt",
        help="Ruta al checkpoint (.pt o .onnx).",
    )
    parser.add_argument(
        "--model-type",
        dest="model_type",
        choices=["pt", "onnx"],
        default="pt",
        help="Tipo de modelo: 'pt' (TwoStreamCNN) o 'onnx'.",
    )
    parser.add_argument(
        "--test-dir",
        dest="test_dir",
        default="data/test_custom",
        help="Directorio con subdirectorios real/ y fake/.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="data/eval_results",
        help="Directorio de salida para CSV, PNG y logs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral de clasificación para P(fake).",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="no_mlflow",
        action="store_true",
        default=False,
        help="Deshabilitar registro en MLflow.",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    checkpoint = Path(args.checkpoint)
    test_dir   = Path(args.test_dir)
    output_dir = Path(args.output_dir)

    # ── Validation ────────────────────────────────────────────────────────────
    if not checkpoint.is_file():
        raise SystemExit(f"[ERROR] Checkpoint no encontrado: '{checkpoint}'")
    if not test_dir.is_dir():
        raise SystemExit(f"[ERROR] test-dir no encontrado: '{test_dir}'")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  VERUM — Evaluación Local del Modelo de Visión")
    print("=" * 65)
    print(f"  Checkpoint  : {checkpoint}")
    print(f"  Modelo tipo : {args.model_type.upper()}")
    print(f"  Test dir    : {test_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Threshold   : {args.threshold}")
    print("=" * 65 + "\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Cargando modelo...")
    if args.model_type == "pt":
        import torch
        device = _select_device()
        model_or_session = _load_pt_model(checkpoint, device)
    else:
        device = None
        model_or_session = _load_onnx_session(checkpoint)

    # ── Collect images ────────────────────────────────────────────────────────
    print("\nRecopilando imágenes...")
    samples = _collect_images(test_dir)
    if not samples:
        raise SystemExit(f"[ERROR] No se encontraron imágenes en '{test_dir}'.")

    n_real = sum(1 for _, l in samples if l == _LABEL_REAL)
    n_fake = sum(1 for _, l in samples if l == _LABEL_FAKE)
    print(f"  Total: {len(samples)} imágenes  (real={n_real}, fake={n_fake})\n")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    print("Evaluando imágenes...")
    t_start = time.monotonic()
    records = _evaluate(samples, args.model_type, model_or_session, device, args.threshold)
    elapsed_total = time.monotonic() - t_start

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\nCalculando métricas globales...")
    metrics = _compute_metrics(records)

    # ── Export CSV ────────────────────────────────────────────────────────────
    csv_path = _export_csv(records, output_dir)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Generando visualizaciones...")
    histogram_path = _save_histogram(records, output_dir)
    cm_path        = _save_confusion_matrix(metrics, output_dir)

    # ── MLflow ────────────────────────────────────────────────────────────────
    if not args.no_mlflow:
        print("Registrando en MLflow...")
        _log_mlflow(metrics, args, histogram_path, cm_path, csv_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    cm = metrics["confusion_matrix"]
    print("\n" + "=" * 65)
    print("  RESULTADOS GLOBALES")
    print("=" * 65)
    print(f"  Imágenes evaluadas : {metrics['valid']}/{metrics['total']}  (errores: {metrics['errors']})")
    print(f"  Tiempo total       : {elapsed_total:.1f}s  ({elapsed_total/max(len(records),1)*1000:.0f}ms/img)")
    print()
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  Recall             : {metrics['recall']:.4f}")
    print(f"  F1                 : {metrics['f1']:.4f}")
    print(f"  AUC-ROC            : {metrics['auc_roc']}")
    print()
    print("  Matriz de confusión (filas=real, columnas=pred):")
    print(f"              Real(0)  Fake(1)")
    print(f"    Real(0)    {cm[0][0]:5d}    {cm[0][1]:5d}   (TN / FP)")
    print(f"    Fake(1)    {cm[1][0]:5d}    {cm[1][1]:5d}   (FN / TP)")
    print()
    print(f"  Salidas guardadas en: {output_dir}/")
    print(f"    results.csv          — resultados imagen por imagen")
    print(f"    score_histogram.png  — distribución de scores")
    print(f"    confusion_matrix.png — matriz de confusión")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
