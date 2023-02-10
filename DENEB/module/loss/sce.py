import math
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SCELoss(t.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = t.nn.CrossEntropyLoss()
        self.A = math.exp(-4)
        
    def forward(self, pred, labels, index=None, mode = None):
        # index is redundant input for SCELoss
        
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = t.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = t.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = t.clamp(label_one_hot, min=self.A, max=1.0)
        rce = (-1*t.sum(pred * t.log(label_one_hot), dim=1))

        # Loss
        if mode == 'ce':
            loss = ce
        else:
            loss = self.alpha * ce + self.beta * rce
        return loss
    