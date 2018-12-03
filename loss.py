import torch
import torch.nn as nn


class L2loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(L2loss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        diff = inputs - targets
        loss = diff.abs().mean()
        # loss = None
        return loss