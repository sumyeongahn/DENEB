import math
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def linear_rampup(current, warm_up, lambda_u, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u*float(current)


class SemiLoss(object):

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up, lambda_u, weight_l=None, weight_u=None):
    
        probs_u = t.softmax(outputs_u, dim=1)

        if weight_l != None and weight_u != None:
            Lx = -t.mean(t.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1) * weight_l)
            Lu = t.mean(t.sum((probs_u - targets_u)**2, dim=1) * weight_u)
        else:
            Lx = -t.mean(t.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1) )
            Lu = t.mean(t.sum((probs_u - targets_u)**2, dim=1))
            
        return Lx, Lu, linear_rampup(epoch,warm_up, lambda_u)
        