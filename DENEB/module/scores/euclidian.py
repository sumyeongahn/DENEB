import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.covariance



class Euclidian(nn.Module):
    def __init__(self,margin=False):
        super(Euclidian, self).__init__()
        self.margin = margin
    
    def forward(self, feat, label):
        feat = feat.squeeze().cuda()
        num_class = int(t.max(label)+1)
        dist = t.zeros((len(feat), num_class))
        for c in range(num_class):
            pos = t.where(label == c)
            center = t.mean(feat[pos], dim=0).cuda()
            dist[:,c] = t.norm(feat - center, p=2, dim = 1).cpu()
        
        given = t.gather(dist, 1 , label.unsqueeze(dim=1)).squeeze()
        if self.margin:
            label_oh = F.one_hot(label, num_classes = num_class)
            label_oh_abs = 1./ t.abs(label_oh-1)
            dist_wo_label = dist * label_oh_abs
            min_wo_given = t.min(dist_wo_label, dim=1)[0]
            return given - min_wo_given
        else:
            return given
        