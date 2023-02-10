import torch as t
import numpy as np

def mixup_data(x, y, alpha=1.0,  device = 'cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
        batch_size = x.size()[0]
        mix_index = t.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[mix_index, :]#
        mixed_target = lam * y + (1 - lam) * y[mix_index, :]


        return mixed_x, mixed_target, lam, mix_index
    else:
        lam = 1
        return x, y, lam, ...

