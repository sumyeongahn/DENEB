import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.covariance


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()
    def forward(self,logit, label):
        b = F.softmax(logit, dim=1) * F.log_softmax(logit, dim=1)
        b = -1.0 * b.sum(dim = 1)
        return b
