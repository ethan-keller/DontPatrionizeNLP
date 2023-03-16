import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logpt = F.log_softmax(output, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, weight=self.weight, ignore_index=self.ignore_index)
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, eps=0.1, weight=None, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.eps = eps
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        loss = loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_preds,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
        )
        return loss


def get_loss_fun(loss_name):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=[0.2, 0.8])
    elif loss_name == "focal":
        return FocalLoss()
    elif loss_name == "label_smoothing":
        return LabelSmoothingLoss()
