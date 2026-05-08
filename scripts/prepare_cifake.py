import shutil
import random
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────────────────
CIFAKE_ROOT = Path("data/raw")  
OUTPUT_ROOT = Path("data/processed")
VAL_RATIO   = 0.15
SEED        = 42
"""
random.seed(SEED)

def split_and_copy(src_dir: Path, split: str, label: str):
    images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    random.shuffle(images)

    if split == "train":
        n_val   = int(len(images) * VAL_RATIO)
        train   = images[n_val:]
        val     = images[:n_val]

        for img in train:
            shutil.copy(img, OUTPUT_ROOT / "train" / label / img.name)
        for img in val:
            shutil.copy(img, OUTPUT_ROOT / "val" / label / img.name)
    else:
        for img in images:
            shutil.copy(img, OUTPUT_ROOT / "test" / label / img.name)

    return len(images)

for split in ["train", "test"]:
    for label in ["REAL", "FAKE"]:
        src = CIFAKE_ROOT / split / label
        out_label = label.lower()
        total = split_and_copy(src, split, out_label)
        print(f"{split}/{label}: {total} imágenes procesadas")

print("\n✓ Dataset organizado en data/processed/")
"""
# Verificación final
for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        count = len(list((OUTPUT_ROOT / split / label).glob("*")))
        print(f"  {split}/{label}: {count} imágenes")