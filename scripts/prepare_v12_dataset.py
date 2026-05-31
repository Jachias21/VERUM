"""
prepare_v12_dataset.py
----------------------
Prepara el dataset v1.2 fusionando tres fuentes:
  1. CIFAKE reducido  — data/processed/train/{real,fake}/   (muestreo aleatorio)
  2. MS COCOAI        — data/processed/mscocoai/{real,fake}/ (todas)
  3. SDXL propias     — data/custom/fake/                    (todas, recursivo)

Estructura de salida:
  data/processed_v12/
    train/
      real/   ← 13 000 CIFAKE-real + todas MS COCOAI-real
      fake/   ← 10 000 CIFAKE-fake + todas MS COCOAI-fake + todas SDXL
    val/      ← copia íntegra de data/processed/val/
    test/     ← copia íntegra de data/processed/test/

Uso:
  python scripts/prepare_v12_dataset.py
"""

import random
import shutil
from pathlib import Path

# ── Reproducibilidad ───────────────────────────────────────────────────────────
random.seed(42)

# ── Rutas base ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

SRC_TRAIN      = REPO_ROOT / "data" / "processed" / "train"
SRC_MSCOCOAI   = REPO_ROOT / "data" / "processed" / "mscocoai"
SRC_CUSTOM_FAKE= REPO_ROOT / "data" / "custom" / "fake"
SRC_VAL        = REPO_ROOT / "data" / "processed" / "val"
SRC_TEST       = REPO_ROOT / "data" / "processed" / "test"

DST_ROOT       = REPO_ROOT / "data" / "processed_v12"
DST_TRAIN_REAL = DST_ROOT / "train" / "real"
DST_TRAIN_FAKE = DST_ROOT / "train" / "fake"
DST_VAL        = DST_ROOT / "val"
DST_TEST       = DST_ROOT / "test"

# ── Cuántas imágenes CIFAKE se usan ───────────────────────────────────────────
N_CIFAKE_REAL = 13_000
N_CIFAKE_FAKE = 10_000

# ── Extensiones válidas ────────────────────────────────────────────────────────
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ── Utilidades ─────────────────────────────────────────────────────────────────

def collect_images(directory: Path, recursive: bool = False) -> list[Path]:
    """Devuelve lista de rutas de imagen dentro de directory."""
    glob = directory.rglob("*") if recursive else directory.glob("*")
    return [
        p for p in glob
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def safe_copy(src: Path, dst_dir: Path, prefix: str = "") -> Path:
    """
    Copia src → dst_dir con resolución de colisiones de nombre.
    Si ya existe el fichero de destino, lo omite (idempotente).
    Devuelve la ruta de destino (haya copiado o no).
    """
    stem = f"{prefix}{src.name}" if prefix else src.name
    dst = dst_dir / stem

    # Colisión de nombre: añade el directorio padre como prefijo
    if dst.exists() and dst.stat().st_size != src.stat().st_size:
        dst = dst_dir / f"{src.parent.name}__{stem}"

    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def copy_all(src_dir: Path, dst_dir: Path, recursive: bool = False,
             label: str = "") -> int:
    """Copia todas las imágenes de src_dir a dst_dir. Devuelve el nº copiadas."""
    if not src_dir.exists():
        print(f"  [AVISO] Directorio no encontrado, se omite: {src_dir}")
        return 0

    images = collect_images(src_dir, recursive=recursive)
    count = 0
    for img in images:
        safe_copy(img, dst_dir)
        count += 1
    tag = f" ({label})" if label else ""
    print(f"  ✓ {count:>6,} imágenes copiadas{tag} ← {src_dir.relative_to(REPO_ROOT)}")
    return count


def copy_sample(src_dir: Path, dst_dir: Path, n: int, label: str = "") -> int:
    """
    Copia una muestra aleatoria de n imágenes de src_dir a dst_dir.
    Si hay menos de n imágenes disponibles, copia todas y avisa.
    """
    if not src_dir.exists():
        print(f"  [AVISO] Directorio no encontrado, se omite: {src_dir}")
        return 0

    images = collect_images(src_dir)
    if len(images) < n:
        print(f"  [AVISO] Solo hay {len(images):,} imágenes en {src_dir.relative_to(REPO_ROOT)} "
              f"(se pedían {n:,}). Se copian todas.")
        sample = images
    else:
        sample = random.sample(images, n)

    count = 0
    for img in sample:
        safe_copy(img, dst_dir)
        count += 1
    tag = f" ({label})" if label else ""
    print(f"  ✓ {count:>6,} imágenes copiadas{tag} ← {src_dir.relative_to(REPO_ROOT)}")
    return count


def count_images(directory: Path) -> int:
    """Cuenta imágenes directamente dentro de directory (no recursivo)."""
    if not directory.exists():
        return 0
    return sum(1 for p in directory.iterdir()
               if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# ── Pipeline principal ─────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'='*62}")
    print("  Dataset v1.2 — Preparación")
    print(f"{'='*62}")
    print(f"  Salida  : {DST_ROOT.relative_to(REPO_ROOT)}")
    print(f"  Seed    : 42")
    print(f"{'='*62}\n")

    # Crear directorios de destino
    for d in (DST_TRAIN_REAL, DST_TRAIN_FAKE):
        d.mkdir(parents=True, exist_ok=True)

    # ── TRAIN / REAL ──────────────────────────────────────────────────────────
    print("── TRAIN / real ─────────────────────────────────────────")
    copy_sample(SRC_TRAIN / "real", DST_TRAIN_REAL,
                n=N_CIFAKE_REAL, label="CIFAKE")
    copy_all(SRC_MSCOCOAI / "real", DST_TRAIN_REAL,
             label="MS COCOAI")

    # ── TRAIN / FAKE ──────────────────────────────────────────────────────────
    print("\n── TRAIN / fake ─────────────────────────────────────────")
    copy_sample(SRC_TRAIN / "fake", DST_TRAIN_FAKE,
                n=N_CIFAKE_FAKE, label="CIFAKE")
    copy_all(SRC_MSCOCOAI / "fake", DST_TRAIN_FAKE,
             label="MS COCOAI")
    copy_all(SRC_CUSTOM_FAKE, DST_TRAIN_FAKE,
             recursive=True, label="SDXL custom")

    # ── VAL — copia íntegra ───────────────────────────────────────────────────
    print("\n── VAL ──────────────────────────────────────────────────")
    if SRC_VAL.exists():
        if DST_VAL.exists():
            shutil.rmtree(DST_VAL)
        shutil.copytree(SRC_VAL, DST_VAL)
        print(f"  ✓ val/ copiado íntegramente ← {SRC_VAL.relative_to(REPO_ROOT)}")
    else:
        print(f"  [AVISO] {SRC_VAL.relative_to(REPO_ROOT)} no encontrado, se omite.")

    # ── TEST — copia íntegra ──────────────────────────────────────────────────
    print("\n── TEST ─────────────────────────────────────────────────")
    if SRC_TEST.exists():
        if DST_TEST.exists():
            shutil.rmtree(DST_TEST)
        shutil.copytree(SRC_TEST, DST_TEST)
        print(f"  ✓ test/ copiado íntegramente ← {SRC_TEST.relative_to(REPO_ROOT)}")
    else:
        print(f"  [AVISO] {SRC_TEST.relative_to(REPO_ROOT)} no encontrado, se omite.")

    # ── RESUMEN FINAL ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  CONTEO FINAL POR SPLIT Y CLASE")
    print(f"{'='*62}")

    splits = [
        ("train", "real", DST_TRAIN_REAL),
        ("train", "fake", DST_TRAIN_FAKE),
    ]

    # val y test pueden tener subcarpetas real/ y fake/
    for split_name, split_dir in [("val", DST_VAL), ("test", DST_TEST)]:
        for cls in ("real", "fake"):
            splits.append((split_name, cls, split_dir / cls))

    total = 0
    for split, cls, path in splits:
        n = count_images(path)
        total += n
        print(f"  {split:5s} / {cls:4s} : {n:>8,}")

    print(f"{'─'*40}")
    print(f"  {'TOTAL':11s}: {total:>8,}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
