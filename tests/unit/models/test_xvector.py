import torch

from voiceid.models.xvector import XVectorEncoder


def test_xvector_output_shape() -> None:
    model = XVectorEncoder(n_mels=80, emb_dim=192)
    x = torch.randn(4, 80, 100)
    y = model(x)
    assert y.shape == (4, 192)
