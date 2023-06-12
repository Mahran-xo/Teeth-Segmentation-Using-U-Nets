import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=0, size_average=None, ignore_index=-100, reduce=None, balance_param=1.0):
        super(FocalLoss, self).__init__(size_average)
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


class MeanDiceScore(nn.Module):
    """ calculates the mean dice score
    """

    def __init__(self, sigmoid=True, weights=None, epsilon=1.e-5):
        super().__init__()

        self.softmax = sigmoid
        self.weights = weights
        self.eps = epsilon

    def forward(self, inputs, targets):

        if self.softmax:
            inputs = torch.sigmoid(inputs)

        if self.weights == None:
            self.weights = torch.ones(inputs.shape[1])
        w = self.weights[None, :, None, None]
        w = w.to(inputs.device)

        num = 2 * torch.sum(inputs * targets * w, dim=(1, 2, 3))
        den = torch.sum((inputs + targets) * w, dim=(1, 2, 3)) + self.eps

        return torch.mean(num / den)


class MeanDiceLoss(nn.Module):
    """ calculates the mean dice loss
    """

    def __init__(self, softmax=True, weights=None, epsilon=1.e-5):
        super().__init__()

        self.dice = MeanDiceScore(softmax, weights, epsilon)

    def forward(self, inputs, targets):
        dice_score = self.dice(inputs, targets)

        return 1 - dice_score


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


class CEDICE(nn.Module):
    def __init__(self):
        super(CEDICE, self).__init__()

    def forward(self, prediction, target, bce_weight=0.5):
        """Calculating the loss and metrics
            Args:
                prediction = predicted image
                target = Targeted image
                metrics = Metrics printed
                bce_weight = 0.5 (default)
            Output:
                loss : dice loss of the epoch """
        bce = F.binary_cross_entropy_with_logits(prediction, target)
        prediction = F.sigmoid(prediction)
        dice = dice_loss(prediction, target)

        loss = bce * bce_weight + dice * (1 - bce_weight)

        return loss


class GeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, inp, targ):
        inp = torch.sigmoid(inp)
        inp = inp.contiguous().permute(0, 2, 3, 1)
        targ = targ.contiguous().permute(0, 2, 3, 1)

        w = torch.zeros((2,))
        w = 1. / (torch.sum(targ, (0, 1, 2)) ** 2 + 1e-9)

        numerator = targ * inp
        numerator = w * torch.sum(numerator, (0, 1, 2))
        numerator = torch.sum(numerator)

        denominator = targ + inp
        denominator = w * torch.sum(denominator, (0, 1, 2))
        denominator = torch.sum(denominator)

        dice = 2. * (numerator + 1e-9) / (denominator + 1e-9)

        return 1. - dice


# PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
