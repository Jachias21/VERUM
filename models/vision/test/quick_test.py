import torch
from architecture import TwoStreamCNN

device = torch.device('cuda')
model = TwoStreamCNN(pretrained=True).to(device)

rgb  = torch.randn(2, 3, 224, 224).to(device)
freq = torch.randn(2, 2, 224, 224).to(device)

out = model(rgb, freq)
print('Forward pass OK:', out.shape)