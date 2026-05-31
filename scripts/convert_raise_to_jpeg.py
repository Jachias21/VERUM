#!/usr/bin/env python3
"""
convert_raise_to_jpeg.py — Convierte imágenes TIFF de RAISE a JPEG.

Lee todos los ficheros .TIF / .tiff de un directorio de entrada,
los convierte a JPEG (calidad 95) con Pillow y los guarda en un
directorio de salida.  Opcionalmente borra el TIFF original tras
convertir.

Uso:
  python scripts/convert_raise_to_jpeg.py \\
      --input-dir  data/raw/raise/images \\
      --output-dir data/raw/raise/jpeg \\
      --max-images 2000 \\
      --delete-originals

Dependencias: Pillow  (pip install Pillow)
"""

import argparse
import sys
import time
from pathlib import Path

# ── Importaciones opcionales con mensajes claros ───────────────────────────────
try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    sys.exit("[ERROR] Instala Pillow:  pip install Pillow")


# ── Constantes ─────────────────────────────────────────────────────────────────
JPEG_QUALITY    = 95          # Calidad JPEG (0-95)
TIFF_EXTENSIONS = {".tif", ".tiff"}  # Extensiones reconocidas (case-insensitive)


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Barra de progreso ASCII."""
    pct  = current / total if total else 0
    done = int(width * pct)
    bar  = "█" * done + "░" * (width - done)
    return f"[{bar}] {current}/{total} ({pct*100:.1f}%)"


def _collect_tiffs(input_dir: Path, max_images: int) -> list[Path]:
    """Recoge hasta max_images ficheros TIFF del directorio (no recursivo)."""
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in TIFF_EXTENSIONS
    )
    return files[:max_images]


def _print_progress(
    counters: dict,
    total: int,
    t_start: float,
) -> None:
    done    = counters["converted"] + counters["skipped"] + counters["errors"]
    bar     = _progress_bar(done, total)
    elapsed = time.time() - t_start
    rate    = done / elapsed if elapsed > 0 else 0
    print(f"  {bar}  {rate:.1f} img/s", end="\r", flush=True)


# ── Conversión principal ───────────────────────────────────────────────────────

def convert(
    input_dir: Path,
    output_dir: Path,
    max_images: int,
    delete_originals: bool,
) -> dict[str, int]:
    """
    Convierte TIFFs a JPEG y los guarda en output_dir.

    Returns:
        dict con contadores {"converted": n, "skipped": n, "errors": n,
                             "deleted": n}
    """
    print(f"\n{'='*62}")
    print("  RAISE — Conversión TIFF → JPEG")
    print(f"{'='*62}")
    print(f"  Entrada          : {input_dir}")
    print(f"  Salida           : {output_dir}")
    print(f"  Límite           : {max_images:,} imágenes")
    print(f"  Calidad JPEG     : {JPEG_QUALITY}")
    print(f"  Borrar originales: {'Sí' if delete_originals else 'No'}")
    print(f"{'='*62}\n")

    # Recoger ficheros fuente
    tiffs = _collect_tiffs(input_dir, max_images)
    if not tiffs:
        print("  ⚠ No se encontraron ficheros TIFF en el directorio de entrada.")
        return {"converted": 0, "skipped": 0, "errors": 0, "deleted": 0}

    total = len(tiffs)
    print(f"  Ficheros TIFF encontrados : {total:,}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    counters = {"converted": 0, "skipped": 0, "errors": 0, "deleted": 0}
    t_start  = time.time()

    for tiff_path in tiffs:
        dest = output_dir / (tiff_path.stem + ".jpg")

        # ── Omitir si el JPEG ya existe y no está vacío ───────────────────────
        if dest.exists() and dest.stat().st_size > 0:
            counters["skipped"] += 1
            _print_progress(counters, total, t_start)
            continue

        # ── Convertir ─────────────────────────────────────────────────────────
        try:
            with Image.open(tiff_path) as img:
                # TIFF puede ser RGBA o con perfil de color; convertir a RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(dest, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            counters["converted"] += 1

            # ── Borrar original si se solicitó ────────────────────────────────
            if delete_originals:
                tiff_path.unlink()
                counters["deleted"] += 1

        except UnidentifiedImageError:
            counters["errors"] += 1
            print(f"\n  ⚠ No se pudo identificar la imagen: {tiff_path.name}")
        except OSError as exc:
            counters["errors"] += 1
            print(f"\n  ⚠ Error al procesar {tiff_path.name}: {str(exc)[:80]}")

        _print_progress(counters, total, t_start)

    return counters


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte imágenes TIFF de RAISE a JPEG (calidad 95).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directorio con los ficheros TIFF de entrada.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directorio donde se guardarán los JPEG convertidos.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=2000,
        metavar="N",
        help="Número máximo de imágenes a convertir.",
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        default=False,
        help="Borra el TIFF original después de convertirlo correctamente.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Validaciones básicas ───────────────────────────────────────────────────
    if not args.input_dir.exists():
        sys.exit(f"[ERROR] El directorio de entrada no existe: {args.input_dir}")
    if not args.input_dir.is_dir():
        sys.exit(f"[ERROR] La ruta de entrada no es un directorio: {args.input_dir}")
    if args.max_images < 1:
        sys.exit("[ERROR] --max-images debe ser >= 1")

    t0       = time.time()
    counters = convert(
        args.input_dir,
        args.output_dir,
        args.max_images,
        args.delete_originals,
    )
    elapsed  = time.time() - t0
    total    = counters["converted"] + counters["skipped"] + counters["errors"]

    # ── Resumen final ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*62}")
    print("  RESUMEN FINAL")
    print(f"{'='*62}")
    print(f"  ✓ Convertidas nuevas    : {counters['converted']:>7,}")
    print(f"  ○ Omitidas (ya existían): {counters['skipped']:>7,}")
    print(f"  ⚠ Errores               : {counters['errors']:>7,}")
    if args.delete_originals:
        print(f"  🗑 Originales borrados  : {counters['deleted']:>7,}")
    print(f"  ─ Total procesadas      : {total:>7,}")
    print(f"  ⏱ Tiempo total          : {elapsed:>6.1f}s")
    if elapsed > 0 and total > 0:
        print(f"  ⚡ Velocidad media       : {total/elapsed:>6.1f} img/s")
    print(f"{'='*62}")
    print(f"  Salida → {args.output_dir.resolve()}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
