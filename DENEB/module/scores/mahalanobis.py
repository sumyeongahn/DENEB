import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.covariance


        
class Mahalanobis(nn.Module):
    def __init__(self, margin=False):
        super(Mahalanobis, self).__init__()
        self.margin = margin
    def forward(self, feat, label):
        feat = feat.squeeze()
        num_class = int(t.max(label)+1)
        dist = t.zeros((len(feat), num_class))

        for c in range(num_class):
            pos = t.where(label == c)
            _feat = feat[pos]
            center = t.mean(_feat, dim=0).cuda()
            centered_feat = _feat - t.mean(_feat, dim=0)
            empirical_cov = sklearn.covariance.EmpiricalCovariance(assume_centered=True)
            empirical_cov.fit(centered_feat)


            inv_cov = t.tensor(empirical_cov.get_precision()).float().cuda()
            _dist = t.zeros(len(feat))

            for sub_c in range(num_class):
                sub_pos = t.where(label == sub_c)
                sub_feat = feat[sub_pos].cuda()
                _dist[sub_pos] = t.sqrt(t.diag(t.mm(t.mm((sub_feat - center), inv_cov) , (sub_feat-center).T ))).cpu()

            dist[:,c] = _dist
        
        if self.margin:
                
            given = t.gather(dist, 1 , label.unsqueeze(dim=1)).squeeze()
            label_oh = F.one_hot(label, num_classes = num_class)
            label_oh_abs = 1./ t.abs(label_oh-1)
            dist_wo_label = dist * label_oh_abs
            min_wo_given = t.min(dist_wo_label, dim=1)[0]
            
            return given - min_wo_given
        else:
            given = t.gather(dist, 1 , label.unsqueeze(dim=1)).squeeze()
            
            return given
            