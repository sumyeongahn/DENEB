import math
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GeneralizedCELoss(nn.Module):
    def __init__(self, reduction=None, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
        self.reduction = reduction
        
    def forward(self, logits, targets):
        if self.q == 0:
            return F.cross_entropy(logits,targets, reduction='none')

        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = t.gather(p, 1, t.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')
        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight


        if self.reduction == 'none':
            return loss
        else:
            return loss.mean()
