import numpy as np
from models.vision.preprocess import rgb_to_freq_tensor, batch_to_freq_tensors
import torch

# Test imagen individual
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
freq = rgb_to_freq_tensor(img)
print('Tensor frecuencial:', freq.shape, '— dtype:', freq.dtype)
print('Rango valores:', round(freq.min().item(), 3), '→', round(freq.max().item(), 3))

# Test batch
batch = torch.randn(4, 3, 224, 224)
freq_batch = batch_to_freq_tensors(batch)
print('Batch frecuencial:', freq_batch.shape)
print('Preprocesado Fourier OK')