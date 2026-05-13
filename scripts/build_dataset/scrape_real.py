"""
scrape_real.py — descarga imágenes reales de Google Images para la clase REAL
del dataset propio de VERUM.

Uso:
    python scripts/build_dataset/scrape_real.py
    python scripts/build_dataset/scrape_real.py --output-dir data/custom/real/ --images-per-query 150

Dependencias:
    pip install icrawler Pillow
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Queries predefinidas agrupadas por categoría
# Las claves son los nombres de subcarpeta; los valores son la lista de
# búsquedas que alimentan esa categoría.
# ---------------------------------------------------------------------------
QUERIES: dict[str, list[str]] = {
    "personas_famosas": [
        "famous politicians portrait photo",
        "celebrities red carpet photo",
        "world leaders meeting press photo",
        "famous musicians live concert photo",
        "athletes sports competition photo",
    ],
    "eventos_sociales": [
        "protest demonstration crowd photo",
        "natural disaster flood earthquake photo",
        "political rally crowd real photo",
        "street festival celebration photo",
        "humanitarian crisis refugee photo",
    ],
    "paisajes_lugares": [
        "mountain landscape nature photo",
        "urban cityscape street photo",
        "ocean beach sunset photo",
        "forest river waterfall photo",
        "desert dunes aerial photo",
    ],
    "animales": [
        "wildlife animals nature photo",
        "dogs cats pets photo",
        "birds flying nature photo",
        "ocean fish marine life photo",
        "farm animals countryside photo",
    ],
    "comida": [
        "restaurant food plated meal photo",
        "fresh fruit vegetables market photo",
        "street food vendor photo",
        "bakery bread pastry photo",
        "traditional cuisine dish photo",
    ],
    "objetos_cotidianos": [
        "everyday objects desk workspace photo",
        "tools hardware workshop photo",
        "electronics gadgets photo",
        "clothing accessories fashion photo",
        "books stationery office photo",
    ],
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_dirname(query: str) -> str:
    """Convert an arbitrary query string to a filesystem-safe directory name."""
    name = query.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name[:60].strip("_")


def _filter_small_images(folder: Path, min_px: int = 256) -> tuple[int, int]:
    """
    Remove images smaller than ``min_px`` in either dimension.

    Returns
    -------
    (kept, removed)
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        log.warning("Pillow no instalado — omitiendo filtrado de tamaño.")
        total = len(list(folder.iterdir()))
        return total, 0

    kept = removed = 0
    for img_path in list(folder.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            if w < min_px or h < min_px:
                img_path.unlink()
                removed += 1
            else:
                kept += 1
        except Exception:  # noqa: BLE001
            # Corrupt / unreadable file — delete it too
            img_path.unlink(missing_ok=True)
            removed += 1

    return kept, removed


def _crawl_query(query: str, dest: Path, max_num: int) -> None:
    """Download up to ``max_num`` images for ``query`` into ``dest``."""
    try:
        from icrawler.builtin import GoogleImageCrawler  # noqa: PLC0415
    except ImportError:
        log.error(
            "icrawler no instalado. Instálalo con:  pip install icrawler"
        )
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)

    crawler = GoogleImageCrawler(
        storage={"root_dir": str(dest)},
        log_level=logging.WARNING,   # supress icrawler verbose output
    )
    crawler.crawl(
        keyword=query,
        max_num=max_num,
        file_idx_offset="auto",     # evita sobrescribir si ya hay imágenes
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga imágenes reales de Google Images para el dataset VERUM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/custom/real"),
        help="Directorio raíz donde se guardan las imágenes descargadas.",
    )
    parser.add_argument(
        "--images-per-query",
        type=int,
        default=100,
        help="Número máximo de imágenes a descargar por cada query.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=256,
        help="Tamaño mínimo (px) en cada dimensión para conservar la imagen.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Pausa en segundos entre queries (evita bloqueos de Google).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_root: Path = args.output_dir
    images_per_query: int = args.images_per_query
    min_size: int = args.min_size
    delay: float = args.delay

    output_root.mkdir(parents=True, exist_ok=True)

    # summary: category → total kept images
    summary: dict[str, int] = {}

    total_categories = len(QUERIES)
    for cat_idx, (category, queries) in enumerate(QUERIES.items(), start=1):
        cat_dir = output_root / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "━━━ Categoría %d/%d: %s (%d queries) ━━━",
            cat_idx, total_categories, category, len(queries),
        )

        for q_idx, query in enumerate(queries, start=1):
            query_dir = cat_dir / _safe_dirname(query)
            log.info(
                "  [%d/%d] Descargando: '%s'  →  %s",
                q_idx, len(queries), query, query_dir,
            )

            _crawl_query(query, query_dir, max_num=images_per_query)

            # Filter small images right after download
            kept, removed = _filter_small_images(query_dir, min_px=min_size)
            log.info(
                "         ✓ %d imágenes conservadas  |  %d eliminadas (< %dpx)",
                kept, removed, min_size,
            )

            if q_idx < len(queries):
                time.sleep(delay)

        # Count total for this category (across all query subfolders)
        cat_total = sum(
            len(list(sub.iterdir()))
            for sub in cat_dir.iterdir()
            if sub.is_dir()
        )
        summary[category] = cat_total

        if cat_idx < total_categories:
            log.info("  Pausa entre categorías (%.1fs)…", delay * 2)
            time.sleep(delay * 2)

    # ── Resumen final ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  RESUMEN — imágenes descargadas por categoría")
    print("═" * 60)
    grand_total = 0
    for category, count in summary.items():
        print(f"  {category:<30s}  {count:>5d} imágenes")
        grand_total += count
    print("─" * 60)
    print(f"  {'TOTAL':<30s}  {grand_total:>5d} imágenes")
    print("═" * 60)
    print(f"\n✓ Imágenes guardadas en: {output_root.resolve()}")


if __name__ == "__main__":
    main()
