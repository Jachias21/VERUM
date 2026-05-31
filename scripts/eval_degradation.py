#!/usr/bin/env python3
"""
eval_degradation.py — Análisis de degradación del modelo de visión VERUM
                      bajo distintas transformaciones de imagen.

Evalúa el mismo test set bajo 5 condiciones aplicadas a cada imagen
antes de la inferencia:

  original   — sin transformación
  jpeg_90    — compresión JPEG calidad 90
  jpeg_70    — compresión JPEG calidad 70  (Telegram estándar)
  jpeg_50    — compresión JPEG calidad 50  (Telegram agresivo)
  resize_160 — redimensionado a 160×160 antes de pasar al modelo

USO:
  python scripts/eval_degradation.py
  python scripts/eval_degradation.py --checkpoint models/vision/weights/verum_cnn_best.pt \\
                                      --test-dir data/test_custom \\
                                      --output-dir data/eval_results/degradation

ESTRUCTURA del directorio de test:
  test_dir/
    real/   ← imágenes reales
    fake/   ← imágenes sintéticas/generadas

SALIDAS (en --output-dir):
  degradation_results.csv    — resultados imagen×condición
  degradation_summary.csv    — tabla resumen Accuracy/F1/AUC por condición
  degradation_curve.png      — curva de degradación JPEG
  degradation_table.png      — tabla visual con métricas comparativas

MLflow:
  Experimento : VERUM-Degradation
  Parámetros  : checkpoint, test_dir, threshold, n_conditions
  Métricas    : accuracy_{cond}, f1_{cond}, auc_roc_{cond} por cada condición
"""
from __future__ import annotations

import argparse
import csv
import io
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
log = logging.getLogger("verum.eval_degradation")

# ── Re-use helpers from eval_local ────────────────────────────────────────────
try:
    from scripts.eval_local import (  # type: ignore[import]
        _load_pt_model,
        _infer_pt,
        _load_rgb,
        _to_spatial_tensor,
        _to_freq_tensor,
        _compute_metrics,
        _collect_images,
        _select_device,
        _LABEL_FAKE,
        _LABEL_REAL,
        _SUPPORTED_EXTS,
    )
except ImportError:
    # Fallback: direct import when running from project root
    from eval_local import (  # type: ignore[import]
        _load_pt_model,
        _infer_pt,
        _load_rgb,
        _to_spatial_tensor,
        _to_freq_tensor,
        _compute_metrics,
        _collect_images,
        _select_device,
        _LABEL_FAKE,
        _LABEL_REAL,
        _SUPPORTED_EXTS,
    )

# ── Conditions ────────────────────────────────────────────────────────────────
CONDITIONS: list[dict[str, Any]] = [
    {"name": "original",   "label": "Original",      "jpeg_quality": None, "resize": None},
    {"name": "jpeg_90",    "label": "JPEG q=90",      "jpeg_quality": 90,   "resize": None},
    {"name": "jpeg_70",    "label": "JPEG q=70",      "jpeg_quality": 70,   "resize": None},
    {"name": "jpeg_50",    "label": "JPEG q=50",      "jpeg_quality": 50,   "resize": None},
    {"name": "resize_160", "label": "Resize 160×160", "jpeg_quality": None, "resize": 160},
]

# JPEG quality values used for the degradation curve plot
_JPEG_CONDITIONS = ["original", "jpeg_90", "jpeg_70", "jpeg_50"]
_JPEG_QUALITY_X  = [100, 90, 70, 50]   # x-axis labels (100 = "original")


# =============================================================================
# Image transformation helpers
# =============================================================================

def _apply_jpeg_compression(img_rgb, quality: int):
    """Round-trip through JPEG at the given quality; returns uint8 RGB array."""
    import cv2
    import numpy as np

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, buf = cv2.imencode(".jpg", img_bgr, encode_params)
    if not ok:
        raise RuntimeError(f"cv2.imencode falló con calidad={quality}")
    img_bgr_decoded = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr_decoded, cv2.COLOR_BGR2RGB)


def _apply_resize(img_rgb, size: int):
    """Resize to size×size and back to original dims; returns uint8 RGB array."""
    import cv2

    h, w = img_rgb.shape[:2]
    small = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    # Restore to original resolution so _to_spatial_tensor gets a natural image
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored


def _transform_image(img_rgb, condition: dict[str, Any]):
    """Apply the condition's transformation and return modified RGB array."""
    transformed = img_rgb

    if condition["resize"] is not None:
        transformed = _apply_resize(transformed, condition["resize"])

    if condition["jpeg_quality"] is not None:
        transformed = _apply_jpeg_compression(transformed, condition["jpeg_quality"])

    return transformed


# =============================================================================
# Per-condition evaluation
# =============================================================================

def _evaluate_condition(
    samples: list[tuple[Path, int]],
    model,
    device,
    condition: dict[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    """
    Run inference on all samples under a single condition.
    Returns per-image records enriched with the condition name.
    """
    records: list[dict[str, Any]] = []
    total    = len(samples)
    cond_name = condition["name"]

    for i, (img_path, label) in enumerate(samples, 1):
        label_str = "fake" if label == _LABEL_FAKE else "real"
        print(f"    [{i:4d}/{total}] {img_path.name:<45}", end="", flush=True)

        t0 = time.monotonic()
        try:
            img_rgb       = _load_rgb(img_path)
            img_rgb_t     = _transform_image(img_rgb, condition)
            score         = _infer_pt(model, img_rgb_t, device)
            pred_label    = _LABEL_FAKE if score >= threshold else _LABEL_REAL
            pred_str      = "fake" if pred_label == _LABEL_FAKE else "real"
            correct       = pred_label == label
            elapsed_ms    = int((time.monotonic() - t0) * 1000)
            error         = None

        except Exception as exc:  # noqa: BLE001
            score      = -1.0
            pred_label = -1
            pred_str   = "ERROR"
            correct    = False
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            error      = str(exc)
            log.warning("Error [%s] '%s': %s", cond_name, img_path.name, exc)

        mark = "✓" if correct else "✗"
        print(f" score={score:.4f}  pred={pred_str:<4}  {mark}  ({elapsed_ms}ms)")

        records.append({
            "condition":      cond_name,
            "filename":       img_path.name,
            "filepath":       str(img_path),
            "label":          label_str,
            "label_int":      label,
            "score":          round(score, 6),
            "predicted":      pred_str,
            "predicted_int":  pred_label,
            "correct":        correct,
            "elapsed_ms":     elapsed_ms,
            "error":          error or "",
        })

    return records


# =============================================================================
# Summary computation
# =============================================================================

def _build_summary(
    all_records: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute per-condition Accuracy / F1 / AUC-ROC summary rows."""
    summary: list[dict[str, Any]] = []

    for cond in conditions:
        cond_records = [r for r in all_records if r["condition"] == cond["name"]]
        try:
            m = _compute_metrics(cond_records)
        except ValueError as exc:
            log.warning("No se pudo calcular métricas para '%s': %s", cond["name"], exc)
            m = {"accuracy": float("nan"), "f1": float("nan"), "auc_roc": "nan",
                 "precision": float("nan"), "recall": float("nan"),
                 "total": len(cond_records), "valid": 0, "errors": len(cond_records)}

        summary.append({
            "condition":  cond["name"],
            "label":      cond["label"],
            "total":      m["total"],
            "valid":      m["valid"],
            "errors":     m["errors"],
            "accuracy":   m["accuracy"],
            "precision":  m["precision"],
            "recall":     m["recall"],
            "f1":         m["f1"],
            "auc_roc":    m["auc_roc"],
        })

    return summary


# =============================================================================
# CSV exports
# =============================================================================

def _export_detail_csv(all_records: list[dict[str, Any]], output_dir: Path) -> Path:
    """Export per-image × per-condition results."""
    out = output_dir / "degradation_results.csv"
    fieldnames = [
        "condition", "filename", "filepath", "label",
        "score", "predicted", "correct", "elapsed_ms", "error",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    log.info("CSV detallado guardado: %s", out)
    return out


def _export_summary_csv(summary: list[dict[str, Any]], output_dir: Path) -> Path:
    """Export per-condition summary metrics."""
    out = output_dir / "degradation_summary.csv"
    fieldnames = ["condition", "label", "total", "valid", "errors",
                  "accuracy", "precision", "recall", "f1", "auc_roc"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    log.info("CSV resumen guardado: %s", out)
    return out


# =============================================================================
# Plots
# =============================================================================

def _save_degradation_curve(summary: list[dict[str, Any]], output_dir: Path) -> Path:
    """
    Line chart showing Accuracy, F1 and AUC-ROC across JPEG quality levels.
    X-axis: quality 100 (original) → 50 (aggressive).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Pull only JPEG-related conditions in order
    jpeg_rows = {r["condition"]: r for r in summary}
    acc_vals  = [jpeg_rows[c]["accuracy"]  for c in _JPEG_CONDITIONS]
    f1_vals   = [jpeg_rows[c]["f1"]        for c in _JPEG_CONDITIONS]
    auc_vals  = []
    for c in _JPEG_CONDITIONS:
        v = jpeg_rows[c]["auc_roc"]
        auc_vals.append(float(v) if v != "nan" else float("nan"))

    x = np.array(_JPEG_QUALITY_X)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x, acc_vals,  "o-",  color="#3498db", linewidth=2.2, markersize=8,
            label="Accuracy",  zorder=3)
    ax.plot(x, f1_vals,   "s--", color="#e74c3c", linewidth=2.2, markersize=8,
            label="F1",        zorder=3)
    ax.plot(x, auc_vals,  "^-.", color="#2ecc71", linewidth=2.2, markersize=8,
            label="AUC-ROC",   zorder=3)

    # Annotate each point
    for xi, acc, f1, auc in zip(x, acc_vals, f1_vals, auc_vals):
        ax.annotate(f"{acc:.3f}", (xi, acc),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=8, color="#3498db", ha="center")
        ax.annotate(f"{f1:.3f}",  (xi, f1),
                    textcoords="offset points", xytext=(0, -15),
                    fontsize=8, color="#e74c3c", ha="center")

    # Shade the degradation region
    ax.axvspan(49, 95, alpha=0.06, color="#e74c3c", label="Zona de compresión")

    ax.set_xlabel("Calidad JPEG  (100 = Original sin compresión)", fontsize=12)
    ax.set_ylabel("Métrica", fontsize=12)
    ax.set_title("Curva de degradación VERUM bajo compresión JPEG", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Original\n(100)", "JPEG 90", "JPEG 70", "JPEG 50"])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    ax.grid(axis="x", alpha=0.20, linestyle=":")
    fig.tight_layout()

    out = output_dir / "degradation_curve.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    log.info("Curva de degradación guardada: %s", out)
    return out


def _save_summary_table_png(summary: list[dict[str, Any]], output_dir: Path) -> Path:
    """Render summary metrics as a styled table PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col_labels = ["Condición", "Total", "Errores", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    rows = []
    for r in summary:
        auc = r["auc_roc"] if r["auc_roc"] == "nan" else f"{float(r['auc_roc']):.4f}"
        rows.append([
            r["label"],
            str(r["total"]),
            str(r["errors"]),
            f"{r['accuracy']:.4f}"  if r["accuracy"]  == r["accuracy"] else "nan",
            f"{r['precision']:.4f}" if r["precision"] == r["precision"] else "nan",
            f"{r['recall']:.4f}"    if r["recall"]    == r["recall"]    else "nan",
            f"{r['f1']:.4f}"        if r["f1"]        == r["f1"]        else "nan",
            auc,
        ])

    # Color rows by Accuracy value (green→red gradient)
    def _row_color(row_data):
        try:
            acc = float(row_data[3])
        except ValueError:
            return "#f5f5f5"
        if acc >= 0.90:
            return "#d5f5e3"
        if acc >= 0.80:
            return "#fef9e7"
        return "#fde8e8"

    cell_colors = [[_row_color(r)] + ["#fafafa"] * (len(col_labels) - 1) for r in rows]

    fig, ax = plt.subplots(figsize=(11, len(rows) * 0.65 + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Header style
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "VERUM — Métricas comparativas por condición de transformación",
        fontsize=13, fontweight="bold", pad=12,
    )
    fig.tight_layout()

    out = output_dir / "degradation_table.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Tabla resumen guardada: %s", out)
    return out


# =============================================================================
# MLflow logging
# =============================================================================

def _log_mlflow(
    summary: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    args: argparse.Namespace,
    curve_path: Path,
    table_path: Path,
    detail_csv: Path,
    summary_csv: Path,
) -> None:
    """Log all results to MLflow under 'VERUM-Degradation'."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow no disponible — saltando registro.")
        return

    mlflow.set_experiment("VERUM-Degradation")
    run_name = f"degradation_{Path(args.checkpoint).stem}"

    with mlflow.start_run(run_name=run_name):
        # Parameters
        mlflow.log_param("checkpoint",   str(args.checkpoint))
        mlflow.log_param("test_dir",     str(args.test_dir))
        mlflow.log_param("threshold",    args.threshold)
        mlflow.log_param("n_conditions", len(CONDITIONS))
        mlflow.log_param("n_images",     len(all_records) // len(CONDITIONS))

        # Per-condition metrics
        for row in summary:
            cond = row["condition"]
            mlflow.log_metric(f"accuracy_{cond}",  row["accuracy"]
                              if row["accuracy"]  == row["accuracy"]  else 0.0)
            mlflow.log_metric(f"f1_{cond}",         row["f1"]
                              if row["f1"]         == row["f1"]         else 0.0)
            if row["auc_roc"] != "nan":
                mlflow.log_metric(f"auc_roc_{cond}", float(row["auc_roc"]))

        # Artifacts
        mlflow.log_artifact(str(curve_path))
        mlflow.log_artifact(str(table_path))
        mlflow.log_artifact(str(detail_csv))
        mlflow.log_artifact(str(summary_csv))

    log.info("Resultados registrados en MLflow (experimento: VERUM-Degradation).")


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análisis de degradación del modelo VERUM bajo transformaciones de imagen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="models/vision/weights/verum_cnn_best.pt",
        help="Ruta al checkpoint .pt (TwoStreamCNN).",
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
        default="data/eval_results/degradation",
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

    print("\n" + "=" * 70)
    print("  VERUM — Análisis de Degradación del Modelo de Visión")
    print("=" * 70)
    print(f"  Checkpoint  : {checkpoint}")
    print(f"  Test dir    : {test_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Threshold   : {args.threshold}")
    print(f"  Condiciones : {', '.join(c['name'] for c in CONDITIONS)}")
    print("=" * 70 + "\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Cargando modelo PT...")
    device = _select_device()
    model  = _load_pt_model(checkpoint, device)

    # ── Collect images ────────────────────────────────────────────────────────
    print("\nRecopilando imágenes del test set...")
    samples = _collect_images(test_dir)
    if not samples:
        raise SystemExit(f"[ERROR] No se encontraron imágenes en '{test_dir}'.")

    n_real = sum(1 for _, l in samples if l == _LABEL_REAL)
    n_fake = sum(1 for _, l in samples if l == _LABEL_FAKE)
    print(f"  Total: {len(samples)} imágenes  (real={n_real}, fake={n_fake})\n")

    # ── Evaluate all conditions ───────────────────────────────────────────────
    all_records: list[dict[str, Any]] = []
    t_global = time.monotonic()

    for idx, condition in enumerate(CONDITIONS, 1):
        print(f"[{idx}/{len(CONDITIONS)}] Condición: {condition['label']}")
        cond_records = _evaluate_condition(samples, model, device, condition, args.threshold)
        all_records.extend(cond_records)
        n_ok = sum(1 for r in cond_records if r["correct"])
        print(f"   → {n_ok}/{len(cond_records)} correctas\n")

    elapsed_total = time.monotonic() - t_global

    # ── Summary metrics ───────────────────────────────────────────────────────
    print("Calculando métricas por condición...")
    summary = _build_summary(all_records, CONDITIONS)

    # ── CSV exports ───────────────────────────────────────────────────────────
    detail_csv  = _export_detail_csv(all_records, output_dir)
    summary_csv = _export_summary_csv(summary, output_dir)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("Generando visualizaciones...")
    curve_path = _save_degradation_curve(summary, output_dir)
    table_path = _save_summary_table_png(summary, output_dir)

    # ── MLflow ────────────────────────────────────────────────────────────────
    if not args.no_mlflow:
        print("Registrando en MLflow (experimento: VERUM-Degradation)...")
        _log_mlflow(summary, all_records, args, curve_path, table_path,
                    detail_csv, summary_csv)

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTADOS COMPARATIVOS POR CONDICIÓN")
    print("=" * 70)
    header = f"  {'Condición':<18} {'Accuracy':>9} {'F1':>9} {'AUC-ROC':>9} {'Errores':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in summary:
        auc = row["auc_roc"] if row["auc_roc"] == "nan" else f"{float(row['auc_roc']):.4f}"
        print(
            f"  {row['label']:<18}"
            f" {row['accuracy']:>9.4f}"
            f" {row['f1']:>9.4f}"
            f" {auc:>9}"
            f" {row['errors']:>8}"
        )
    print()
    print(f"  Tiempo total : {elapsed_total:.1f}s")
    print(f"  Imágenes     : {len(samples)} × {len(CONDITIONS)} condiciones"
          f" = {len(all_records)} inferencias")
    print()
    print(f"  Salidas guardadas en: {output_dir}/")
    print(f"    degradation_results.csv  — resultados imagen × condición")
    print(f"    degradation_summary.csv  — métricas por condición")
    print(f"    degradation_curve.png    — curva de degradación JPEG")
    print(f"    degradation_table.png    — tabla visual comparativa")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
