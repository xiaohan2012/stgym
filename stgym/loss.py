import torch
import torch.nn.functional as F


def compute_classification_loss(pred: torch.Tensor, true: torch.Tensor):
    """Compute loss and prediction score.

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Ground truth labels

    Returns: Loss, normalized prediction score

    """
    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    # multiclass
    if pred.ndim > 1 and true.ndim == 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true), pred
    else:
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        # binary or multilabel
        true = true.float()
        return bce_loss(pred, true), torch.sigmoid(pred)
