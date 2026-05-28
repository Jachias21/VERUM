#!/usr/bin/env python3
"""
eval_nlp.py — Evaluación cuantitativa del módulo NLP de VERUM contra un gold set.

Ejecuta el pipeline completo (NER → hybrid_search → synthesize_verdict →
_extract_verdict_from_llm_output) para cada ejemplo del gold set y genera:

  reports/eval_summary.json  — métricas en formato máquina-legible
  reports/eval_report.html   — informe HTML con tabla, matriz de confusión y fallos

USO:
  python scripts/eval_nlp.py --gold tests/golden/gold_nlp.jsonl --out reports/

REQUISITOS del entorno:
  - Qdrant corriendo y accesible (QDRANT_HOST / QDRANT_PORT)
  - Ollama corriendo con el modelo configurado en OLLAMA_MODEL
  - Las mismas variables de entorno que usa worker_nlp en producción
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# ── Ensure project root is in sys.path ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Register bare app.* aliases for worker_nlp so its internal imports work
import importlib as _importlib

_nlp_app = _importlib.import_module("services.worker_nlp.app")
sys.modules.setdefault("app", _nlp_app)
for _name in ("cache", "metrics", "ner", "rag"):
    sys.modules.setdefault(f"app.{_name}", _importlib.import_module(f"services.worker_nlp.app.{_name}"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("verum.eval_nlp")

# ── Pipeline imports (after path setup) ───────────────────────────────────────
from services.worker_nlp.app.ner import extract_entities, is_gibberish  # noqa: E402
from services.worker_nlp.app.rag import (  # noqa: E402
    hybrid_search,
    synthesize_verdict,
    _extract_verdict_from_llm_output,
    _topic_overlap_score,
)

LABELS = ["FAKE", "REAL", "UNVERIFIED"]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

async def _run_one(example: dict[str, Any]) -> dict[str, Any]:
    """Run the full NLP pipeline on a single gold example. Never raises."""
    text: str = example["text"]
    qid = uuid.uuid4()
    t0 = time.monotonic()

    try:
        # Short-circuit: gibberish detection (mirrors worker.py logic)
        if is_gibberish(text):
            latency_ms = int((time.monotonic() - t0) * 1000)
            return {
                "id": example["id"],
                "expected": example["expected_verdict"],
                "predicted": "UNVERIFIED",
                "latency_ms": latency_ms,
                "source_url": None,
                "topic_overlap": None,
                "category": example.get("category", ""),
                "error": None,
            }

        entities = extract_entities(text)
        rag_result = await hybrid_search(qid, text, entities)

        # Calculate topic overlap for the winning hit (if any)
        topic_overlap: float | None = None
        if rag_result.retrieved_context:
            topic_overlap = _topic_overlap_score(entities, rag_result.retrieved_context)

        rag_result = await synthesize_verdict(rag_result, text)

        # LLM verdict override (same logic as worker.py)
        llm_verdict = _extract_verdict_from_llm_output(rag_result.summary)
        predicted = llm_verdict if llm_verdict is not None else rag_result.verdict

        latency_ms = int((time.monotonic() - t0) * 1000)
        return {
            "id": example["id"],
            "expected": example["expected_verdict"],
            "predicted": predicted,
            "latency_ms": latency_ms,
            "source_url": rag_result.source_url,
            "topic_overlap": topic_overlap,
            "category": example.get("category", ""),
            "error": None,
        }

    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.error("Error processing example %s: %s", example["id"], exc)
        return {
            "id": example["id"],
            "expected": example["expected_verdict"],
            "predicted": "ERROR",
            "latency_ms": latency_ms,
            "source_url": None,
            "topic_overlap": None,
            "category": example.get("category", ""),
            "error": str(exc),
        }


async def _run_all(gold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run examples sequentially to avoid overwhelming Qdrant/Ollama."""
    results = []
    total = len(gold)
    for i, example in enumerate(gold, 1):
        print(f"  [{i:3d}/{total}] {example['id']} ...", end="", flush=True)
        result = await _run_one(example)
        status = "OK" if result["error"] is None else "ERR"
        match = "✓" if result["predicted"] == result["expected"] else "✗"
        print(f" {match} {result['predicted']:<12} ({result['latency_ms']}ms) [{status}]")
        results.append(result)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute precision/recall/F1, confusion matrix and latency percentiles."""
    try:
        import numpy as np
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            precision_recall_fscore_support,
        )
    except ImportError as e:
        raise SystemExit(f"scikit-learn or numpy not available: {e}") from e

    valid = [r for r in results if r["predicted"] != "ERROR"]
    y_true = [r["expected"] for r in valid]
    y_pred = [r["predicted"] for r in valid]

    # Map unknown predictions (e.g. model hallucination) to UNVERIFIED
    y_pred_clean = [p if p in LABELS else "UNVERIFIED" for p in y_pred]

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred_clean, labels=LABELS, zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    cm = confusion_matrix(y_true, y_pred_clean, labels=LABELS).tolist()

    per_class: dict[str, dict] = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    # Latency percentiles
    latencies = [r["latency_ms"] for r in results]
    lat_arr = np.array(latencies)
    lat_by_class: dict[str, dict] = {}
    for label in LABELS:
        cls_lat = [r["latency_ms"] for r in results if r["expected"] == label]
        if cls_lat:
            a = np.array(cls_lat)
            lat_by_class[label] = {
                "mean_ms": round(float(a.mean()), 1),
                "p50_ms": round(float(np.percentile(a, 50)), 1),
                "p95_ms": round(float(np.percentile(a, 95)), 1),
            }

    # URL coverage
    url_count = sum(1 for r in valid if r["source_url"])
    url_pct = round(url_count / len(valid) * 100, 1) if valid else 0.0

    # Accuracy per category
    categories: dict[str, dict] = {}
    for r in valid:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["predicted"] == r["expected"]:
            categories[cat]["correct"] += 1
    cat_accuracy = {
        cat: round(v["correct"] / v["total"], 4) if v["total"] else 0.0
        for cat, v in categories.items()
    }

    return {
        "total_examples": len(results),
        "errors": sum(1 for r in results if r["error"] is not None),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": LABELS,
        "latency_mean_ms": round(float(lat_arr.mean()), 1),
        "latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 1),
        "latency_by_class": lat_by_class,
        "url_coverage_pct": url_pct,
        "accuracy_by_category": cat_accuracy,
        "detail": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix image (base64 PNG)
# ─────────────────────────────────────────────────────────────────────────────

def _cm_png_b64(cm: list[list[int]]) -> str:
    """Render confusion matrix as a base64-encoded PNG string."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        return ""

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm),
        display_labels=LABELS,
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Matriz de confusión — VERUM NLP")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _render_html(metrics: dict[str, Any], cm_b64: str) -> str:
    cm_img_tag = (
        f'<img src="data:image/png;base64,{cm_b64}" alt="Confusion matrix" style="max-width:480px;">'
        if cm_b64
        else "<p><em>matplotlib no disponible — imagen omitida.</em></p>"
    )

    # Build per-class table rows
    class_rows = ""
    for label in LABELS:
        pc = metrics["per_class"].get(label, {})
        class_rows += (
            f"<tr><td><b>{label}</b></td>"
            f"<td>{pc.get('precision', 0):.4f}</td>"
            f"<td>{pc.get('recall', 0):.4f}</td>"
            f"<td>{pc.get('f1', 0):.4f}</td>"
            f"<td>{pc.get('support', 0)}</td></tr>\n"
        )

    # Build category accuracy rows
    cat_rows = ""
    for cat, acc in sorted(metrics["accuracy_by_category"].items()):
        cat_rows += f"<tr><td>{cat}</td><td>{acc:.2%}</td></tr>\n"

    # Build failure list
    failures = [r for r in metrics["detail"] if r["predicted"] != r["expected"]]
    fail_rows = ""
    for r in failures:
        truncated = (r.get("id", "") + " — see gold set")
        err = r.get("error") or ""
        fail_rows += (
            f"<tr>"
            f"<td>{r['id']}</td>"
            f"<td>{r['expected']}</td>"
            f"<td>{r['predicted']}</td>"
            f"<td>{r['category']}</td>"
            f"<td>{r['latency_ms']}ms</td>"
            f"<td style='color:#c00'>{err}</td>"
            f"</tr>\n"
        )
    if not fail_rows:
        fail_rows = "<tr><td colspan='6' style='text-align:center'>Sin fallos</td></tr>"

    macro_color = "green" if metrics["macro_f1"] >= 0.75 else "red"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VERUM NLP — Informe de Evaluación</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 2em auto; color: #222; }}
  h1 {{ color: #1a3c6e; }}
  h2 {{ color: #2c5f9e; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5em; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
  th {{ background: #e8f0fe; text-align: center; }}
  td:first-child {{ text-align: left; }}
  .macro {{ font-size: 2em; font-weight: bold; color: {macro_color}; }}
  .pass {{ color: green; font-weight: bold; }}
  .fail {{ color: red; font-weight: bold; }}
  .section {{ margin-bottom: 2em; }}
</style>
</head>
<body>
<h1>VERUM NLP — Informe de Evaluación</h1>

<div class="section">
<h2>Resumen</h2>
<table>
  <tr><th>Métrica</th><th>Valor</th></tr>
  <tr><td>Ejemplos totales</td><td>{metrics['total_examples']}</td></tr>
  <tr><td>Errores de pipeline</td><td>{metrics['errors']}</td></tr>
  <tr><td>Cobertura con URL fuente</td><td>{metrics['url_coverage_pct']}%</td></tr>
  <tr><td>Latencia media</td><td>{metrics['latency_mean_ms']} ms</td></tr>
  <tr><td>Latencia p50</td><td>{metrics['latency_p50_ms']} ms</td></tr>
  <tr><td>Latencia p95</td><td>{metrics['latency_p95_ms']} ms</td></tr>
</table>
<p>Macro-F1 (objetivo ≥ 0.75):
  <span class="macro">{metrics['macro_f1']:.4f}</span>
  {'<span class="pass">✓ SUPERA EL UMBRAL</span>' if metrics['macro_f1'] >= 0.75 else '<span class="fail">✗ NO ALCANZA EL UMBRAL</span>'}
</p>
</div>

<div class="section">
<h2>Métricas por clase</h2>
<table>
  <tr><th>Clase</th><th>Precisión</th><th>Recall</th><th>F1</th><th>Soporte</th></tr>
  {class_rows}
</table>
</div>

<div class="section">
<h2>Matriz de confusión</h2>
{cm_img_tag}
</div>

<div class="section">
<h2>Accuracy por categoría</h2>
<table>
  <tr><th>Categoría</th><th>Accuracy</th></tr>
  {cat_rows}
</table>
</div>

<div class="section">
<h2>Latencia por clase</h2>
<table>
  <tr><th>Clase</th><th>Media (ms)</th><th>p50 (ms)</th><th>p95 (ms)</th></tr>
  {"".join(
    f"<tr><td>{lbl}</td><td>{v['mean_ms']}</td><td>{v['p50_ms']}</td><td>{v['p95_ms']}</td></tr>"
    for lbl, v in metrics.get("latency_by_class", {}).items()
  )}
</table>
</div>

<div class="section">
<h2>Predicciones incorrectas ({len(failures)})</h2>
<table>
  <tr><th>ID</th><th>Esperado</th><th>Predicho</th><th>Categoría</th><th>Latencia</th><th>Error</th></tr>
  {fail_rows}
</table>
</div>

</body>
</html>
"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación cuantitativa del módulo NLP de VERUM contra un gold set."
    )
    parser.add_argument(
        "--gold",
        default="tests/golden/gold_nlp.jsonl",
        help="Ruta al gold set JSONL (default: tests/golden/gold_nlp.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="reports/",
        help="Directorio de salida para los informes (default: reports/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar a los primeros N ejemplos (útil para pruebas rápidas)",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise SystemExit(f"Gold set no encontrado: {gold_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load gold set
    gold: list[dict[str, Any]] = []
    with gold_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                gold.append(json.loads(line))

    if args.limit:
        gold = gold[: args.limit]

    print(f"\nVERUM NLP — Evaluación contra gold set")
    print(f"  Gold set : {gold_path} ({len(gold)} ejemplos)")
    print(f"  Salida   : {out_dir}")
    print()

    # Run pipeline
    results = await _run_all(gold)

    # Compute metrics
    print("\nCalculando métricas...")
    metrics = _compute_metrics(results)

    # Write JSON summary
    summary_path = out_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2, default=str)
    print(f"  [OK] {summary_path}")

    # Generate HTML report
    print("Generando informe HTML...")
    cm_b64 = _cm_png_b64(metrics["confusion_matrix"])
    html = _render_html(metrics, cm_b64)
    report_path = out_dir / "eval_report.html"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  [OK] {report_path}")

    # Print summary to stdout
    print()
    print("=" * 50)
    print(f"  Macro-F1      : {metrics['macro_f1']:.4f}  (objetivo ≥ 0.75)")
    print(f"  Errores       : {metrics['errors']}/{metrics['total_examples']}")
    for label in LABELS:
        pc = metrics["per_class"].get(label, {})
        print(f"  {label:<12}: P={pc.get('precision', 0):.3f}  R={pc.get('recall', 0):.3f}  F1={pc.get('f1', 0):.3f}")
    print(f"  Latencia p50  : {metrics['latency_p50_ms']} ms")
    print(f"  Latencia p95  : {metrics['latency_p95_ms']} ms")
    print("=" * 50)

    if metrics["macro_f1"] >= 0.75:
        print("  ✓ Macro-F1 supera el umbral de 0.75 definido en el TFM.")
    else:
        print("  ✗ Macro-F1 no alcanza el umbral de 0.75. Revisar pipeline.")
    print()


if __name__ == "__main__":
    asyncio.run(_main())
