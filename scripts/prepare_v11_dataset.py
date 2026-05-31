"""
prepare_v11_dataset.py
----------------------
Copia recursivamente todas las imágenes de data/custom/fake/ hacia
data/processed/train/fake/ usando shutil.copy2.

Al finalizar imprime:
  - Cuántas imágenes se copiaron en esta ejecución.
  - El total de imágenes que hay en train/fake/ (incluyendo las preexistentes).
"""

import shutil
from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
SRC_DIR     = REPO_ROOT / "data" / "custom" / "fake"
DST_DIR     = REPO_ROOT / "data" / "processed" / "train" / "fake"

# Extensiones reconocidas como imagen
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# ── Preparación ────────────────────────────────────────────────────────────────
DST_DIR.mkdir(parents=True, exist_ok=True)

if not SRC_DIR.exists():
    raise FileNotFoundError(f"El directorio fuente no existe: {SRC_DIR}")

# ── Copia ──────────────────────────────────────────────────────────────────────
copied = 0
skipped = 0

for src_path in SRC_DIR.rglob("*"):
    if src_path.is_file() and src_path.suffix.lower() in IMAGE_EXTS:
        dst_path = DST_DIR / src_path.name

        # Si ya existe un archivo con ese nombre, añade el sufijo del padre
        # para evitar colisiones entre subcarpetas distintas.
        if dst_path.exists():
            dst_path = DST_DIR / f"{src_path.parent.name}__{src_path.name}"

        if dst_path.exists():
            skipped += 1
            continue

        shutil.copy2(src_path, dst_path)
        copied += 1

# ── Resumen ────────────────────────────────────────────────────────────────────
total_in_dst = sum(
    1 for f in DST_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in IMAGE_EXTS
)

print(f"✅  Imágenes copiadas  : {copied}")
print(f"⏭️   Omitidas (ya existían): {skipped}")
print(f"📂  Total en train/fake : {total_in_dst}")
