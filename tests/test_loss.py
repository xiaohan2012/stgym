import torch
from stgym.loss import compute_classification_loss

def test_compute_classification_loss():
    true = torch.tensor([0, 1, 0, 1])
    pred = torch.tensor([.5, .5, .5, .5])
    value, pred_ = compute_classification_loss(pred, true)
    assert isinstance(value, torch.Tensor)
    assert value.ndim == 0
    assert isinstance(pred_, torch.Tensor)
    assert pred_.shape == pred.shape







