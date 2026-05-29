"""
Arquitectura CNN de dos flujos para detección de imágenes generadas por IA.

Rama espacial   → procesa la imagen RGB original.
Rama frecuencial → procesa el espectro DFT high-pass de los canales Cb/Cr.
Ambas ramas se fusionan por concatenación antes de la cabeza de clasificación.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class FrequencyBranch(nn.Module):
    """CNN ligero para el tensor de espectro frecuencial de 2 canales (Cb, Cr)."""

    def __init__(self, out_features: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(128 * 4 * 4, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.fc(x.flatten(1))


class TwoStreamCNN(nn.Module):
    """
    Clasificador de dos ramas.
      spatial_branch : backbone EfficientNet-B0 (preentrenado en ImageNet).
      freq_branch    : FrequencyBranch personalizado.
      head           : MLP → sigmoid → P(sintético).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Rama espacial - sustituir el clasificador final
        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        self.spatial_branch = nn.Sequential(*list(efficientnet.children())[:-1])
        spatial_out = 1280  # tamaño de características de EfficientNet-B0

        self.freq_branch = FrequencyBranch(out_features=512)

        self.head = nn.Sequential(
            nn.Linear(spatial_out + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        s = self.spatial_branch(rgb).flatten(1)
        f = self.freq_branch(freq)
        return self.head(torch.cat([s, f], dim=1))
