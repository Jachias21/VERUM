#!/usr/bin/env python3
"""
pipeline_monitor.py — Monitorización del pipeline de visión y calibración de threshold.

Parte 1 — Monitorización: traza el viaje de una imagen por el pipeline paso a paso.
Parte 2 — Calibración: calcula el threshold óptimo (Youden + F1) sobre el test set.

USO:
  # Sólo monitorización
  python scripts/pipeline_monitor.py --checkpoint models/vision/weights/verum_cnn_best.pt \
      --monitor-image path/to/img.jpg

  # Sólo calibración
  python scripts/pipeline_monitor.py --checkpoint models/vision/weights/verum_cnn_best.pt \
      --test-dir data/test_custom

  # Ambas
  python scripts/pipeline_monitor.py --checkpoint models/vision/weights/verum_cnn_best.pt \
      --monitor-image path/to/img.jpg --test-dir data/test_custom

SALIDAS (en --output-dir, default data/eval_results/monitor):
  original.png            — imagen original
  fourier_cb.png          — espectro Fourier canal Cb
  fourier_cr.png          — espectro Fourier canal Cr
  pipeline_report.txt     — informe texto con estadísticas paso a paso
  roc_curve.png           — curva ROC con thresholds óptimos marcados
  threshold_metrics.csv   — métricas por threshold
  calibration_report.txt  — informe de calibración

MLflow: experimento VERUM-ThresholdCalibration
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("verum.pipeline_monitor")

# Reutilizamos helpers de eval_local
from scripts.eval_local import (  # noqa: E402
    _load_pt_model,
    _load_rgb,
    _select_device,
    _to_spatial_tensor,
    _to_freq_tensor,
    _collect_images,
    _SUPPORTED_EXTS,
    _LABEL_FAKE,
    _LABEL_REAL,
)

_IMG_SIZE = 224


# =============================================================================
# PARTE 1 — Monitorización del pipeline
# =============================================================================

def _channel_stats(arr) -> dict[str, float]:
    """Devuelve mean, std, min, max de un array numpy."""
    import numpy as np
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "min":  float(np.min(arr)),
        "max":  float(np.max(arr)),
    }


def _dft_magnitude(channel_f32):
    """Aplica DFT + high-pass filter; devuelve el espectro (magnitud log) y estadísticas."""
    import numpy as np
    import cv2

    h, w = channel_f32.shape
    dft = cv2.dft(channel_f32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # High-pass filter: elimina las frecuencias bajas del centro
    mask = np.ones((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    r = max(h, w) // 10  # radio del filtro paso-alto
    for y in range(h):
        for x in range(w):
            if (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2:
                mask[y, x] = 0.0
    mask3 = np.stack([mask, mask], axis=-1)
    dft_hp = dft_shift * mask3

    magnitude = cv2.magnitude(dft_hp[:, :, 0], dft_hp[:, :, 1])
    log_magnitude = np.log1p(magnitude)

    energy = float(np.sum(magnitude ** 2))
    stats = _channel_stats(log_magnitude)
    stats["energy"] = energy
    return log_magnitude, stats


def _save_spectrum_png(spectrum, title: str, out_path: Path) -> None:
    """Guarda el espectro como PNG normalizado con colormap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    norm = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(norm, cmap="inferno", origin="upper")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info("Espectro guardado: %s", out_path)


def _save_original_png(img_rgb, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_rgb)
    ax.set_title("Imagen Original", fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info("Imagen original guardada: %s", out_path)


def run_monitor(args, model, device, output_dir: Path) -> None:
    """Parte 1: monitoriza el viaje de una imagen por el pipeline."""
    import numpy as np
    import cv2

    img_path = Path(args.monitor_image)
    if not img_path.is_file():
        raise SystemExit(f"[ERROR] Imagen no encontrada: '{img_path}'")

    lines = []
    def log_section(title: str):
        sep = "=" * 60
        lines.append(f"\n{sep}\n  {title}\n{sep}")
        print(f"\n{'='*60}\n  {title}\n{'='*60}")

    def log_line(label: str, value: Any):
        s = f"  {label:<40} {value}"
        lines.append(s)
        print(s)

    print("\n" + "=" * 60)
    print("  VERUM — Monitorización del Pipeline de Visión")
    print("=" * 60)
    print(f"  Imagen: {img_path}")
    print(f"  Output: {output_dir}\n")

    # ── PASO 0: Cargar imagen original ───────────────────────────────────────
    log_section("PASO 0 — Imagen Original (RGB)")
    img_rgb = _load_rgb(img_path)
    h_orig, w_orig = img_rgb.shape[:2]
    log_line("Resolución (H x W):", f"{h_orig} x {w_orig}")
    log_line("Canales:", 3)

    for i, canal in enumerate(["R", "G", "B"]):
        s = _channel_stats(img_rgb[:, :, i])
        log_line(f"  {canal} — mean/std/min/max:",
                 f"{s['mean']:.2f} / {s['std']:.2f} / {s['min']:.0f} / {s['max']:.0f}")

    _save_original_png(img_rgb, output_dir / "original.png")

    # ── PASO 1: Conversión a YCbCr ───────────────────────────────────────────
    log_section("PASO 1 — Conversión a YCbCr")
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)  # OpenCV usa YCrCb
    # Canales: Y=0, Cr=1, Cb=2  →  reordenamos a Y, Cb, Cr
    Y  = img_ycbcr[:, :, 0].astype(np.float32)
    Cr = img_ycbcr[:, :, 1].astype(np.float32)
    Cb = img_ycbcr[:, :, 2].astype(np.float32)

    for canal, arr in [("Y (luminancia)", Y), ("Cb (chroma-blue)", Cb), ("Cr (chroma-red)", Cr)]:
        s = _channel_stats(arr)
        log_line(f"  {canal} — mean/std/min/max:",
                 f"{s['mean']:.2f} / {s['std']:.2f} / {s['min']:.0f} / {s['max']:.0f}")

    # ── PASO 2: DFT + High-Pass Filter sobre Cb y Cr ────────────────────────
    log_section("PASO 2 — DFT + High-Pass Filter (Cb y Cr)")

    # Redimensionar a 224 antes del DFT (igual que el modelo)
    Cb_r = cv2.resize(Cb, (_IMG_SIZE, _IMG_SIZE))
    Cr_r = cv2.resize(Cr, (_IMG_SIZE, _IMG_SIZE))

    spec_cb, stats_cb = _dft_magnitude(Cb_r)
    spec_cr, stats_cr = _dft_magnitude(Cr_r)

    for canal, stats in [("Cb", stats_cb), ("Cr", stats_cr)]:
        log_line(f"  {canal} espectro — mean/std:", f"{stats['mean']:.4f} / {stats['std']:.4f}")
        log_line(f"  {canal} espectro — min/max:",  f"{stats['min']:.4f} / {stats['max']:.4f}")
        log_line(f"  {canal} energía high-pass:",   f"{stats['energy']:.2f}")

    _save_spectrum_png(spec_cb, "Espectro Fourier — Canal Cb (high-pass)", output_dir / "fourier_cb.png")
    _save_spectrum_png(spec_cr, "Espectro Fourier — Canal Cr (high-pass)", output_dir / "fourier_cr.png")

    # ── PASO 3: Inferencia del modelo ────────────────────────────────────────
    log_section("PASO 3 — Inferencia del Modelo (TwoStreamCNN)")
    import torch
    rgb_t  = _to_spatial_tensor(img_rgb).to(device)
    freq_t = _to_freq_tensor(img_rgb).to(device)

    t0 = time.monotonic()
    with torch.no_grad():
        score = float(model(rgb_t, freq_t).item())
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    threshold = args.threshold
    verdict   = "FAKE" if score >= threshold else "REAL"

    log_line("Score P(fake):", f"{score:.6f}")
    log_line("Threshold:",     f"{threshold}")
    log_line("Veredicto:",     verdict)
    log_line("Tiempo inferencia:", f"{elapsed_ms} ms")

    # ── Guardar informe ──────────────────────────────────────────────────────
    report_path = output_dir / "pipeline_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Informe del pipeline guardado: %s", report_path)

    print(f"\n  ✓ Visualizaciones guardadas en: {output_dir}/")
    print(f"    original.png   — imagen original")
    print(f"    fourier_cb.png — espectro Fourier Cb")
    print(f"    fourier_cr.png — espectro Fourier Cr")
    print(f"    pipeline_report.txt — informe completo")


# =============================================================================
# PARTE 2 — Calibración del threshold
# =============================================================================

def _score_dataset(samples, model, device) -> tuple[list[float], list[int]]:
    """Infiere scores para todas las muestras del dataset."""
    import torch
    scores, labels = [], []
    total = len(samples)
    print(f"\n  Inferencia sobre {total} imágenes...")
    for i, (img_path, label) in enumerate(samples, 1):
        print(f"  [{i:4d}/{total}] {img_path.name:<45}", end="", flush=True)
        try:
            img_rgb = _load_rgb(img_path)
            rgb_t  = _to_spatial_tensor(img_rgb).to(device)
            freq_t = _to_freq_tensor(img_rgb).to(device)
            with torch.no_grad():
                sc = float(model(rgb_t, freq_t).item())
            scores.append(sc)
            labels.append(label)
            print(f" score={sc:.4f}")
        except Exception as exc:
            log.warning("Error en '%s': %s", img_path.name, exc)
            print(f" ERROR: {exc}")
    return scores, labels


def _threshold_metrics(y_true, y_scores, threshold: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import numpy as np
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    return {
        "threshold": threshold,
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }


def _save_roc_curve(
    fpr, tpr, thresholds_roc, auc_val: float,
    youden_thr: float, f1_thr: float,
    current_thr: float, output_dir: Path
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#3498db", lw=2.5, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#bdc3c7", lw=1.2, linestyle="--", label="Random")

    # Marcar threshold Youden
    idx_y = np.argmin(np.abs(thresholds_roc - youden_thr))
    ax.plot(fpr[idx_y], tpr[idx_y], "o", color="#e74c3c", markersize=10,
            label=f"Youden thr={youden_thr:.3f}")

    # Marcar threshold F1 óptimo
    idx_f = np.argmin(np.abs(thresholds_roc - f1_thr))
    ax.plot(fpr[idx_f], tpr[idx_f], "s", color="#2ecc71", markersize=10,
            label=f"F1-óptimo thr={f1_thr:.3f}")

    # Marcar threshold actual
    idx_c = np.argmin(np.abs(thresholds_roc - current_thr))
    ax.plot(fpr[idx_c], tpr[idx_c], "^", color="#f39c12", markersize=10,
            label=f"Actual thr={current_thr:.2f}")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Curva ROC — VERUM Vision\nCalibración de Threshold", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = output_dir / "roc_curve.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info("Curva ROC guardada: %s", out_path)
    return out_path


def run_calibration(args, model, device, output_dir: Path) -> None:
    """Parte 2: calibración del threshold óptimo sobre el test set."""
    import numpy as np
    from sklearn.metrics import roc_curve, roc_auc_score, f1_score

    test_dir = Path(args.test_dir)
    samples  = _collect_images(test_dir)
    if not samples:
        raise SystemExit(f"[ERROR] No se encontraron imágenes en '{test_dir}'.")

    print("\n" + "=" * 60)
    print("  VERUM — Calibración del Threshold Óptimo")
    print("=" * 60)
    print(f"  Test dir : {test_dir}")
    print(f"  Output   : {output_dir}")

    scores, labels = _score_dataset(samples, model, device)
    y_true   = np.array(labels)
    y_scores = np.array(scores)

    # ── Curva ROC ─────────────────────────────────────────────────────────────
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    auc_val = float(roc_auc_score(y_true, y_scores))

    # ── Threshold óptimo: Youden (max TPR - FPR) ──────────────────────────────
    youden_idx = int(np.argmax(tpr - fpr))
    youden_thr = float(thresholds_roc[youden_idx])

    # ── Threshold óptimo: F1 máximo ───────────────────────────────────────────
    f1_scores = []
    thr_grid  = np.linspace(0.01, 0.99, 199)
    for t in thr_grid:
        y_pred = (y_scores >= t).astype(int)
        f1_scores.append(float(f1_score(y_true, y_pred, zero_division=0)))
    best_f1_idx = int(np.argmax(f1_scores))
    f1_thr      = float(thr_grid[best_f1_idx])
    best_f1_val = f1_scores[best_f1_idx]

    print(f"\n  AUC-ROC             : {auc_val:.4f}")
    print(f"  Threshold Youden    : {youden_thr:.4f}  "
          f"(TPR={tpr[youden_idx]:.4f}, FPR={fpr[youden_idx]:.4f})")
    print(f"  Threshold F1-óptimo : {f1_thr:.4f}  (F1={best_f1_val:.4f})")
    print(f"  Threshold actual    : {args.threshold}")

    # ── Curva ROC PNG ─────────────────────────────────────────────────────────
    roc_path = _save_roc_curve(
        fpr, tpr, thresholds_roc, auc_val,
        youden_thr, f1_thr, args.threshold, output_dir
    )

    # ── Tabla de métricas por threshold ───────────────────────────────────────
    thr_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = [_threshold_metrics(y_true.tolist(), y_scores.tolist(), t) for t in thr_list]

    csv_path = output_dir / "threshold_metrics.csv"
    fieldnames = ["threshold", "accuracy", "f1", "precision", "recall"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV de métricas guardado: %s", csv_path)

    print("\n  Tabla de métricas por threshold:")
    print(f"  {'Thr':>6}  {'Accuracy':>9}  {'F1':>8}  {'Precision':>10}  {'Recall':>8}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*8}")
    for r in rows:
        marker = " ← actual" if r["threshold"] == args.threshold else ""
        print(f"  {r['threshold']:>6.2f}  {r['accuracy']:>9.4f}  "
              f"{r['f1']:>8.4f}  {r['precision']:>10.4f}  {r['recall']:>8.4f}{marker}")

    # ── Informe texto ─────────────────────────────────────────────────────────
    report_lines = [
        "VERUM — Informe de Calibración del Threshold",
        "=" * 60,
        f"Test dir           : {test_dir}",
        f"Total imágenes     : {len(samples)}",
        f"AUC-ROC            : {auc_val:.4f}",
        "",
        f"Threshold Youden   : {youden_thr:.4f}",
        f"  TPR              : {tpr[youden_idx]:.4f}",
        f"  FPR              : {fpr[youden_idx]:.4f}",
        f"  J (TPR-FPR)      : {tpr[youden_idx] - fpr[youden_idx]:.4f}",
        "",
        f"Threshold F1-óptimo: {f1_thr:.4f}",
        f"  F1               : {best_f1_val:.4f}",
        "",
        "Métricas por threshold:",
        f"{'Thr':>6}  {'Accuracy':>9}  {'F1':>8}  {'Precision':>10}  {'Recall':>8}",
    ]
    for r in rows:
        report_lines.append(
            f"{r['threshold']:>6.2f}  {r['accuracy']:>9.4f}  "
            f"{r['f1']:>8.4f}  {r['precision']:>10.4f}  {r['recall']:>8.4f}"
        )

    rep_path = output_dir / "calibration_report.txt"
    rep_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("Informe de calibración guardado: %s", rep_path)

    # ── MLflow ────────────────────────────────────────────────────────────────
    if not args.no_mlflow:
        _log_mlflow_calibration(
            auc_val, youden_thr, f1_thr, best_f1_val,
            tpr[youden_idx], fpr[youden_idx],
            args, roc_path, csv_path, rep_path
        )

    print(f"\n  ✓ Salidas guardadas en: {output_dir}/")
    print(f"    roc_curve.png          — curva ROC con thresholds")
    print(f"    threshold_metrics.csv  — métricas por threshold")
    print(f"    calibration_report.txt — informe completo")


def _log_mlflow_calibration(
    auc_val, youden_thr, f1_thr, best_f1,
    tpr_y, fpr_y,
    args, roc_path, csv_path, rep_path
) -> None:
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow no disponible — saltando registro.")
        return

    mlflow.set_experiment("VERUM-ThresholdCalibration")
    with mlflow.start_run(run_name=f"calibration_{Path(args.checkpoint).stem}"):
        mlflow.log_param("checkpoint",        str(args.checkpoint))
        mlflow.log_param("test_dir",          str(args.test_dir))
        mlflow.log_param("threshold_current", args.threshold)

        mlflow.log_metric("auc_roc",          auc_val)
        mlflow.log_metric("threshold_youden", youden_thr)
        mlflow.log_metric("tpr_at_youden",    tpr_y)
        mlflow.log_metric("fpr_at_youden",    fpr_y)
        mlflow.log_metric("youden_j",         tpr_y - fpr_y)
        mlflow.log_metric("threshold_f1_opt", f1_thr)
        mlflow.log_metric("f1_at_opt",        best_f1)

        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(rep_path))

    log.info("Resultados registrados en MLflow (experimento: VERUM-ThresholdCalibration).")


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitorización del pipeline de visión VERUM y calibración de threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Ruta al checkpoint .pt (TwoStreamCNN).",
    )
    parser.add_argument(
        "--monitor-image",
        dest="monitor_image",
        default=None,
        help="Imagen concreta a monitorizar (Parte 1).",
    )
    parser.add_argument(
        "--test-dir",
        dest="test_dir",
        default=None,
        help="Directorio con real/ y fake/ para calibración (Parte 2).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="data/eval_results/monitor",
        help="Directorio de salida.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold actual para comparar.",
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

    if args.monitor_image is None and args.test_dir is None:
        raise SystemExit(
            "[ERROR] Debes especificar al menos --monitor-image o --test-dir.\n"
            "  Parte 1 (monitorización) requiere --monitor-image\n"
            "  Parte 2 (calibración)    requiere --test-dir"
        )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"[ERROR] Checkpoint no encontrado: '{checkpoint}'")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  VERUM — Pipeline Monitor & Threshold Calibration")
    print("=" * 60)
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Output dir : {output_dir}")
    print(f"  Threshold  : {args.threshold}")

    # Cargar modelo (compartido entre Parte 1 y Parte 2)
    print("\nCargando modelo PT...")
    device = _select_device()
    model  = _load_pt_model(checkpoint, device)

    # ── Parte 1 ───────────────────────────────────────────────────────────────
    if args.monitor_image:
        run_monitor(args, model, device, output_dir)

    # ── Parte 2 ───────────────────────────────────────────────────────────────
    if args.test_dir:
        run_calibration(args, model, device, output_dir)

    print("\n" + "=" * 60)
    print("  ✓ Pipeline Monitor completado.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
