import torch
print('PyTorch:', torch.__version__)
print('MPS disponible:', torch.backends.mps.is_available())
x = torch.randn(1, 3, 224, 224)
if torch.backends.mps.is_available():
    x = x.to('mps')
    print('Forward pass en MPS:', x.shape)
else:
    print('Usando CPU')