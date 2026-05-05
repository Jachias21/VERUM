"""
Evaluation script — computes accuracy, F1, AUC and exports model to ONNX.

Usage:
  python models/vision/evaluate.py --checkpoint weights/verum_cnn.pt
  python models/vision/evaluate.py --checkpoint weights/verum_cnn.pt --export-onnx
"""
import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from architecture import TwoStreamCNN
from train import get_transforms


def evaluate(checkpoint: Path, data_root: Path, export_onnx: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStreamCNN(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    test_ds = datasets.ImageFolder(data_root / "test", transform=get_transforms(train=False))
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(device)
            freq = torch.zeros(imgs.size(0), 2, 224, 224).to(device)  # TODO: real freq tensor
            scores = model(imgs, freq).squeeze(1).cpu()
            preds = (scores >= 0.5).int()
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_scores.extend(scores.tolist())

    print(f"Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1       : {f1_score(all_labels, all_preds):.4f}")
    print(f"AUC      : {roc_auc_score(all_labels, all_scores):.4f}")

    if export_onnx:
        dummy_rgb  = torch.randn(1, 3, 224, 224).to(device)
        dummy_freq = torch.randn(1, 2, 224, 224).to(device)
        onnx_path = checkpoint.with_suffix(".onnx")
        torch.onnx.export(
            model, (dummy_rgb, dummy_freq), onnx_path,
            input_names=["rgb", "freq"], output_names=["score"],
            opset_version=17,
        )
        print(f"ONNX model exported to {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=Path, required=True)
    parser.add_argument("--data-root",    type=Path, default=Path("../../data/processed"))
    parser.add_argument("--export-onnx",  action="store_true")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_root, args.export_onnx)
