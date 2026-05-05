"""
Training script for TwoStreamCNN.

Dataset expected layout:
  data/processed/
    train/real/<images>
    train/fake/<images>
    val/real/<images>
    val/fake/<images>

Usage:
  python models/vision/train.py --epochs 30 --batch-size 32
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from architecture import TwoStreamCNN


def get_transforms(train: bool) -> transforms.Compose:
    ops = [transforms.Resize((224, 224))]
    if train:
        ops += [
            # Simulate Telegram JPEG compression (key augmentation)
            transforms.RandomApply([transforms.Lambda(lambda img: _jpeg_compress(img, quality=60))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


def _jpeg_compress(img, quality: int = 70):
    """Apply JPEG compression artifact simulation."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def train(epochs: int, batch_size: int, lr: float, data_root: Path, output_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStreamCNN(pretrained=True).to(device)

    train_ds = datasets.ImageFolder(data_root / "train", transform=get_transforms(train=True))
    val_ds   = datasets.ImageFolder(data_root / "val",   transform=get_transforms(train=False))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
            # TODO: generate freq tensor from imgs (call preprocessing)
            freq = torch.zeros(imgs.size(0), 2, 224, 224).to(device)  # placeholder
            preds = model(imgs, freq)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch}/{epochs} — loss: {loss.item():.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "verum_cnn.pt")
    print(f"Model saved to {output_dir}/verum_cnn.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--data-root",  type=Path,  default=Path("../../data/processed"))
    parser.add_argument("--output-dir", type=Path,  default=Path("weights"))
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.lr, args.data_root, args.output_dir)
