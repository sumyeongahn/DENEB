import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.covariance

        
class CosSim(nn.Module):
    def __init__(self, eigen=False):
        super(CosSim, self).__init__()
        self.eigen = eigen
    
    def forward(self, feat, label):
        feat = feat.squeeze().cuda()
        num_class = int(t.max(label)+1)
        sim = t.zeros((len(feat), num_class))
        
        if self.eigen:
            
            for c in range(num_class):
                pos = t.where(label == c)
                _feat = feat[pos].cpu()
                _,_,avg_feat = np.linalg.svd(_feat)
                sim[:,c] = F.cosine_similarity(t.tensor(avg_feat[0]).unsqueeze(dim=0).cuda(), feat)
            
            given = t.gather(sim, 1, label.unsqueeze(dim=1)).squeeze()
            return given
        else:
            for c in range(num_class):
                pos = t.where(label == c)
                _feat = feat[pos].cpu()
                avg_feat = t.mean(_feat, dim=0).cuda()
                sim[:,c] = F.cosine_similarity(avg_feat.unsqueeze(dim=0), feat).cpu()
            
            given = t.gather(sim, 1, label.unsqueeze(dim=1)).squeeze()
            return given
