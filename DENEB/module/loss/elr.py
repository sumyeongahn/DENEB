import math
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def cross_entropy(output, target, M=3):
    return F.cross_entropy(output, target)

class elr_plus_loss(nn.Module):
    def __init__(self, num_examp, config, device, num_classes=10, beta=0.3):
        super(elr_plus_loss, self).__init__()
        self.config = config
        self.pred_hist = (t.zeros(num_examp, num_classes)).to(device)
        self.q = 0
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, iteration, output, y_labeled, weight = None):
        y_pred = F.softmax(output,dim=1)
        y_pred = t.clamp(y_pred, 1e-4, 1.0-1e-4)

        if self.num_classes == 100:
            y_labeled = y_labeled*self.q
            y_labeled = y_labeled/(y_labeled).sum(dim=1,keepdim=True)

        if weight == None:
            _ce_loss = t.mean(-t.sum(y_labeled * F.log_softmax(output, dim=1), dim = -1))
        else:
            _ce_loss = t.mean(-t.sum(y_labeled * F.log_softmax(output, dim=1), dim = -1)* weight)
        reg = ((1-(self.q * y_pred).sum(dim=1)).log()).mean()
        final_loss = _ce_loss + sigmoid_rampup(iteration, int(self.config.coef_step))*(int(self.config.lbd) * reg)

        return  final_loss, y_pred.cpu().detach()

    def update_hist(self, epoch, out, index= None, mix_index = ..., mixup_l = 1):
        y_pred_ = F.softmax(out,dim=1)
        self.pred_hist[index] = self.beta * self.pred_hist[index] +  (1-self.beta) *  y_pred_/(y_pred_).sum(dim=1,keepdim=True)
        self.q = mixup_l * self.pred_hist[index]  + (1-mixup_l) * self.pred_hist[index][mix_index]

