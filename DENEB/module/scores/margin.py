import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.covariance



class Margin(nn.Module):
    def __init__(self):
        super(Margin, self).__init__()
    def forward(self,logit,label):
        logit = logit.squeeze()
        # num_class = int(t.max(label)+1)
        num_class = logit.shape[1]
        given = t.gather(logit, 1, label.unsqueeze(dim=1)).squeeze()
        label_oh = F.one_hot(label, num_classes = num_class)
        label_oh_abs = t.abs(label_oh-1)
        logit_wo_label = logit * label_oh_abs
        max_wo_given = t.max(logit_wo_label, dim=1)[0]

        return given - max_wo_given
