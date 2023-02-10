import math
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class OneHotCELoss(t.nn.Module):
    def __init__(self,reduction='none'):
        super(OneHotCELoss, self).__init__()
        self.reduction=False if reduction == 'none' else True

    def forward(self, pred, labels):
        log_softmax = F.log_softmax(pred, dim=1)
        ret = -1.0 * t.sum(labels * log_softmax, dim=1)

        if self.reduction:
            return ret.mean()
        else:
            return ret

        
    