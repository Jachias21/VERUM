"""
Training script for TwoStreamCNN.

Dataset expected layout:
  data/processed/
    train/real/<images>
    train/fake/<images>
    val/real/<images>
    val/fake/<images>

Usage:
  python models/vision/train_v13.py --epochs 10 --batch-size 32
  python models/vision/train_v13.py --epochs 10 --mlflow-uri ./mlruns
  python models/vision/train_v13.py --resume-from weights/verum_cnn_best.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from architecture import TwoStreamCNN
from preprocess import batch_to_freq_tensors

# ---------------------------------------------------------------------------
# MLflow — opcional; el script funciona sin él
# ---------------------------------------------------------------------------
try:
    import mlflow
    import mlflow.pytorch
    _mlflow_available = True
except ImportError:
    _mlflow_available = False


def _jpeg_compress_wrapper(img):
    return _jpeg_compress(img, quality=60)


def get_transforms(train: bool) -> transforms.Compose:
    ops = [transforms.Resize((224, 224))]
    if train:
        ops += [
            transforms.RandomApply(
                [transforms.Lambda(_jpeg_compress_wrapper)], p=0.5
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


def _jpeg_compress(img, quality: int = 70):
    """Simula la compresión JPEG que aplica Telegram a las imágenes."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(
    epochs: int,
    batch_size: int,
    lr: float,
    data_root: Path,
    output_dir: Path,
    mlflow_uri: str = "./mlruns",
    resume_from: Path = None,
):
    device = get_device()
    print(f"[train] Usando dispositivo: {device}")

    # ── MLflow run ───────────────────────────────────────────────────────────
    _mlflow_enabled = _mlflow_available
    if _mlflow_enabled:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("VERUM-TwoStreamCNN")
        mlflow.start_run()
        mlflow.log_params({
            "epochs":        epochs,
            "batch_size":    batch_size,
            "lr":            lr,
            "arquitectura":  "TwoStreamCNN-EfficientNetB0",
            "dataset":       "CIFAKE",
        })
        print(f"[train] MLflow tracking activado → {mlflow_uri}")
    else:
        print("[train] mlflow no instalado — tracking desactivado.")

    model = TwoStreamCNN(pretrained=True).to(device)

    # ── Cargar pesos desde checkpoint si se especifica --resume-from ─────────
    if resume_from is not None:
        print(f"[train] Cargando pesos desde checkpoint: {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=device, weights_only=True))
        print("[train] Checkpoint cargado correctamente.")

    # ── Fine-tuning: congelar backbone espacial inicialmente ─────────────────
    for name, param in model.spatial_branch.named_parameters():
        param.requires_grad = False
    print("[train] Backbone EfficientNet congelado — solo se entrenan head y rama frecuencial")

    train_ds = datasets.ImageFolder(data_root / "train", transform=get_transforms(train=True))
    val_ds   = datasets.ImageFolder(data_root / "val",   transform=get_transforms(train=False))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Verificar que las clases están en el orden correcto (fake=0, real=1)
    print(f"[train] Clases detectadas: {train_ds.class_to_idx}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):

        # ── Descongelar backbone en epoch 3 ──────────────────────────────────
        if epoch == 3:
            for param in model.spatial_branch.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr / 10, weight_decay=1e-4)
            print("[train] Epoch 3 — backbone descongelado, fine-tuning completo con lr reducido")

        # ── Training loop ─────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch:02d}/{epochs} [Train]"):
            imgs   = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            freq = batch_to_freq_tensors(imgs).to(device)

            preds = model(imgs, freq)
            loss  = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            train_correct += ((preds >= 0.5) == labels.bool()).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        # ── Validation loop ───────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_dl, desc=f"Epoch {epoch:02d}/{epochs} [Val]"):
                imgs   = imgs.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                freq   = batch_to_freq_tensors(imgs).to(device)

                preds = model(imgs, freq)
                val_correct += ((preds >= 0.5) == labels.bool()).sum().item()
                val_total   += imgs.size(0)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total
        avg_loss  = train_loss    / train_total

        print(
            f"Epoch {epoch:02d}/{epochs} — "
            f"loss: {avg_loss:.4f} — "
            f"train_acc: {train_acc:.4f} — "
            f"val_acc: {val_acc:.4f}"
        )

        # ── MLflow métricas por epoch ─────────────────────────────────────────
        if _mlflow_enabled:
            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "train_acc":  train_acc,
                    "val_acc":    val_acc,
                },
                step=epoch,
            )

        # Guardar el mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir.mkdir(parents=True, exist_ok=True)
            best_pt = output_dir / "verum_cnn_best.pt"
            torch.save(model.state_dict(), best_pt)
            print(f"  ✓ Mejor modelo guardado (val_acc: {val_acc:.4f})")
            if _mlflow_enabled:
                mlflow.log_artifact(str(best_pt))

    # Guardar checkpoint final
    final_pt = output_dir / "verum_cnn_final.pt"
    torch.save(model.state_dict(), final_pt)
    print(f"\n[train] Entrenamiento completado. Mejor val_acc: {best_val_acc:.4f}")

    if _mlflow_enabled:
        mlflow.log_artifact(str(final_pt))
        mlflow.end_run()
        print(f"[train] MLflow run cerrado. Resultados en: {mlflow_uri}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--data-root",   type=Path,  default=Path("../../data/processed"))
    parser.add_argument("--output-dir",  type=Path,  default=Path("weights"))
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="./mlruns",
        help="URI del servidor MLflow (default: ./mlruns local). "
             "Ejemplo remoto: http://mlflow-server:5000",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Ruta a un checkpoint .pt desde el que reanudar el entrenamiento. "
             "Los pesos se cargan justo después de crear el modelo y antes de congelar el backbone.",
    )
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_root=args.data_root,
        output_dir=args.output_dir,
        mlflow_uri=args.mlflow_uri,
        resume_from=args.resume_from,
    )
