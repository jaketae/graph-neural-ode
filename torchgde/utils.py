import torch
import torch.nn as nn
import torch.nn.functional as F


class PerformanceContainer:
    """Simple data class for metrics logging."""

    def __init__(self, data: dict):
        self.data = data

    @staticmethod
    def deep_update(x, y):
        for key in y.keys():
            x.update({key: list(x[key] + y[key])})
        return x


def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    """Standard percentage accuracy computation."""
    preds = torch.max(y_hat, 1)[1]
    return torch.mean((y == preds).float())


class MAPELoss(nn.Module):
    def forward(self, estimation: torch.Tensor, target: torch.Tensor):
        return torch.abs((target - estimation) / (target + 1e-10))  # Absolute error ratio


class MAELoss(nn.Module):
    def forward(self, estimation: torch.Tensor, target: torch.Tensor):
        AE = torch.abs(target - estimation)
        MAE = AE.mean()
        return MAE
