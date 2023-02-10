import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.covariance




class inner(nn.Module):
    def __init__(self):
        super(inner, self).__init__()
    
    def forward(self, feat, label):
        feat = feat.squeeze()
        num_class = int(t.max(label)+1)
        inner = t.zeros((len(feat), num_class))
        feat = feat.cuda()
        for c in range(num_class):
            pos = t.where(label == c)
            _feat = feat[pos].cpu()
            _,_,avg_feat = np.linalg.svd(_feat)
            avg_feat = t.tensor(avg_feat[0])
            avg_feat = avg_feat.repeat(len(feat),1).cuda()
            inner[:,c] = t.sum(feat* avg_feat, dim=1).cpu()
        
        given = t.gather(inner, 1, label.unsqueeze(dim=1)).squeeze()
        return given
