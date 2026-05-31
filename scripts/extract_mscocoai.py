#!/usr/bin/env python3
"""
extract_mscocoai.py — Extrae imágenes del dataset MS COCOAI desde archivos Parquet.

Organiza las imágenes en:
  <output-dir>/real/   ← label 0
  <output-dir>/fake/   ← label 1

Uso:
  python scripts/extract_mscocoai.py \
      --input-dir  data/raw/mscocoai \
      --output-dir data/processed/mscocoai \
      --max-per-class 5000

Dependencias: pandas, pyarrow, Pillow
"""

import argparse
import io
import sys
import time
from pathlib import Path

# ── Importaciones opcionales con mensajes de error amigables ──────────────────
try:
    import pandas as pd
except ImportError:
    sys.exit("[ERROR] Instala pandas:   pip install pandas pyarrow")

try:
    from PIL import Image
except ImportError:
    sys.exit("[ERROR] Instala Pillow:   pip install Pillow")


# ── Constantes ────────────────────────────────────────────────────────────────
LABEL_COLUMN_CANDIDATES = [
    "label_A", "label", "Label_A", "Label", "Label_B", "label_B", "class", "is_fake", "fake",
]
IMAGE_COLUMN_CANDIDATES = ["image", "img", "pixel_values", "bytes", "Image", "Img"]
LABEL_REAL = 0
LABEL_FAKE = 1


# ── Utilidades ────────────────────────────────────────────────────────────────

def _find_column(df_columns: list[str], candidates: list[str], kind: str) -> str:
    """Devuelve el primer candidato que exista en df_columns."""
    for c in candidates:
        if c in df_columns:
            return c
    raise ValueError(
        f"No se encontró columna de {kind}. "
        f"Columnas disponibles: {df_columns}. "
        f"Candidatos esperados: {candidates}"
    )


def _bytes_to_pil(value) -> Image.Image:
    """Convierte bytes, dict HuggingFace o PIL.Image a PIL.Image."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        # Formato HuggingFace datasets: {"bytes": b"...", "path": "..."}
        raw = value.get("bytes") or value.get("path")
        if isinstance(raw, bytes):
            return Image.open(io.BytesIO(raw)).convert("RGB")
        raise ValueError(f"Dict de imagen sin clave 'bytes' válida: {list(value.keys())}")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise TypeError(f"Tipo de imagen no soportado: {type(value)}")


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Barra de progreso ASCII sencilla."""
    pct = current / total if total else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct*100:.1f}%)"


# ── Lógica principal ──────────────────────────────────────────────────────────

def extract(
    input_dir: Path,
    output_dir: Path,
    max_per_class: int,
    image_format: str = "JPEG",
) -> dict[str, int]:
    """
    Lee todos los .parquet de input_dir y guarda las imágenes en
    output_dir/real/ y output_dir/fake/.

    Returns:
        dict con contadores {"real": n, "fake": n, "errors": n, "skipped": n}
    """
    parquet_files = sorted(input_dir.glob("**/*.parquet"))
    if not parquet_files:
        sys.exit(f"[ERROR] No se encontraron archivos .parquet en: {input_dir}")

    print(f"\n{'='*60}")
    print(f"  MS COCOAI — Extracción de imágenes")
    print(f"{'='*60}")
    print(f"  Entrada : {input_dir}")
    print(f"  Salida  : {output_dir}")
    print(f"  Archivos: {len(parquet_files)} .parquet encontrados")
    print(f"  Límite  : {max_per_class:,} imágenes por clase")
    print(f"{'='*60}\n")

    # Crear directorios de salida
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    counters = {"real": 0, "fake": 0, "errors": 0, "skipped": 0}
    img_col = None
    lbl_col = None
    t_start = time.time()

    for file_idx, pq_path in enumerate(parquet_files, 1):
        print(f"[{file_idx}/{len(parquet_files)}] Procesando: {pq_path.name}")

        try:
            df = pd.read_parquet(pq_path)
        except Exception as exc:
            print(f"  ⚠ No se pudo leer el archivo: {exc}")
            counters["errors"] += 1
            continue

        # Detectar columnas en el primer archivo válido
        if img_col is None or lbl_col is None:
            cols = df.columns.tolist()
            try:
                img_col = _find_column(cols, IMAGE_COLUMN_CANDIDATES, "imagen")
                lbl_col = _find_column(cols, LABEL_COLUMN_CANDIDATES, "etiqueta")
            except ValueError as exc:
                sys.exit(f"[ERROR] {exc}")
            print(f"  Columna imagen  : '{img_col}'")
            print(f"  Columna etiqueta: '{lbl_col}'")

        n_rows = len(df)
        for row_idx, (_, row) in enumerate(df.iterrows()):
            # Verificar si ya alcanzamos el límite en ambas clases
            if counters["real"] >= max_per_class and counters["fake"] >= max_per_class:
                print("  ✓ Límite por clase alcanzado. Deteniendo extracción.")
                break

            label_val = row[lbl_col]

            # Normalizar etiqueta (soporta 0/1, True/False, "real"/"fake", etc.)
            try:
                label_int = int(label_val)
            except (ValueError, TypeError):
                lv = str(label_val).strip().lower()
                label_int = LABEL_FAKE if lv in {"fake", "1", "true", "generated", "ai"} else LABEL_REAL

            # Determinar destino
            if label_int == LABEL_REAL:
                if counters["real"] >= max_per_class:
                    counters["skipped"] += 1
                    continue
                dest_dir = real_dir
                cls_key = "real"
            elif label_int == LABEL_FAKE:
                if counters["fake"] >= max_per_class:
                    counters["skipped"] += 1
                    continue
                dest_dir = fake_dir
                cls_key = "fake"
            else:
                # Etiqueta desconocida: se omite
                counters["skipped"] += 1
                continue

            # Convertir y guardar imagen
            try:
                pil_img = _bytes_to_pil(row[img_col])
                global_idx = counters["real"] + counters["fake"]
                filename = f"{cls_key}_{global_idx:07d}.jpg"
                pil_img.save(dest_dir / filename, format=image_format, quality=95)
                counters[cls_key] += 1
            except Exception as exc:
                counters["errors"] += 1
                if counters["errors"] <= 10:  # evitar flood de mensajes
                    print(f"  ⚠ Fila {row_idx}: {exc}")
                continue

            # Progreso cada 200 imágenes
            total_done = counters["real"] + counters["fake"]
            if total_done % 200 == 0 and total_done > 0:
                bar = _progress_bar(total_done, max_per_class * 2)
                elapsed = time.time() - t_start
                rate = total_done / elapsed if elapsed > 0 else 0
                print(f"  {bar}  {rate:.0f} img/s", end="\r", flush=True)

        else:
            # El loop completó todas las filas sin break
            total_done = counters["real"] + counters["fake"]
            bar = _progress_bar(total_done, max_per_class * 2)
            print(f"  {bar}", flush=True)
            continue

        # Si hubo break (límite alcanzado), salir también del loop de archivos
        break

    return counters


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae imágenes del dataset MS COCOAI desde archivos Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directorio que contiene los archivos .parquet del dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directorio de salida; se crearán subdirectorios real/ y fake/.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=5000,
        metavar="N",
        help="Número máximo de imágenes a extraer por clase (real/fake).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="JPEG",
        choices=["JPEG", "PNG", "WEBP"],
        help="Formato de imagen de salida.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        sys.exit(f"[ERROR] El directorio de entrada no existe: {args.input_dir}")

    if args.max_per_class < 1:
        sys.exit("[ERROR] --max-per-class debe ser >= 1")

    t0 = time.time()
    counters = extract(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_per_class=args.max_per_class,
        image_format=args.format,
    )
    elapsed = time.time() - t0

    # ── Resumen final ─────────────────────────────────────────────────────────
    total = counters["real"] + counters["fake"]
    print(f"\n{'='*60}")
    print("  RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  ✓ Imágenes reales guardadas : {counters['real']:>8,}")
    print(f"  ✓ Imágenes fake  guardadas  : {counters['fake']:>8,}")
    print(f"  ─ Total guardadas            : {total:>8,}")
    print(f"  Errores de decodificación : {counters['errors']:>8,}")
    print(f"  Omitidas (límite/label)   : {counters['skipped']:>8,}")
    print(f"  Tiempo total              : {elapsed:>7.1f}s")
    if elapsed > 0:
        print(f"  ⚡ Velocidad media           : {total/elapsed:>7.0f} img/s")
    print(f"{'='*60}")
    print(f"  Salida → {args.output_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
