import torch

from voiceid.losses.aam_softmax import AAMSoftmaxLoss


def test_aam_softmax_returns_scalar_loss() -> None:
    loss_fn = AAMSoftmaxLoss(emb_dim=192, num_classes=10)
    emb = torch.randn(8, 192)
    labels = torch.randint(0, 10, (8,))
    loss = loss_fn(emb, labels)
    assert loss.ndim == 0
