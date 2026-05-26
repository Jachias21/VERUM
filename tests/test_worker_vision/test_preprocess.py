import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from models.vision.preprocess import rgb_to_freq_tensor, batch_to_freq_tensors


def test_rgb_to_freq_tensor_shape_and_dtype():
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    freq = rgb_to_freq_tensor(img)
    assert freq.shape == (2, 224, 224)
    assert freq.dtype == torch.float32


def test_rgb_to_freq_tensor_range():
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    freq = rgb_to_freq_tensor(img)
    assert freq.min().item() >= 0.0
    assert freq.max().item() <= 1.0


def test_batch_to_freq_tensors_shape():
    batch = torch.randn(4, 3, 224, 224)
    freq_batch = batch_to_freq_tensors(batch)
    assert freq_batch.shape == (4, 2, 224, 224)