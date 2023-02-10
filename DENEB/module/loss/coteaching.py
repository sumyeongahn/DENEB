import math
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


    
def coteaching_loss(y_1, y_2, target, forget_rate, weight = None):
    if weight == None:
        loss_1 = F.cross_entropy(y_1, target, reduction = 'none')
    else:
        loss_1 = F.cross_entropy(y_1, target, reduction = 'none') * weight
    ind_1_sorted = t.argsort(loss_1).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    if weight == None:
        loss_2 = F.cross_entropy(y_2, target, reduction = 'none')
    else:
        loss_2 = F.cross_entropy(y_2, target, reduction = 'none') * weight
    ind_2_sorted = t.argsort(loss_2).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], target[ind_2_update], reduction = 'none')
    loss_2_update = F.cross_entropy(y_2[ind_1_update], target[ind_1_update], reduction = 'none')

    return t.sum(loss_1_update)/num_remember, t.sum(loss_2_update)/num_remember, ind_1_update, ind_2_update, loss_1, loss_2

def coteaching_plus_loss(logits, logits2, labels, ind, forget_rate, step, weight = None):
    _, pred1 = t.max(logits.data, 1)
    _, pred2 = t.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id=np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    
    temp_disagree = ind*logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0]==len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]

    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(t.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = logits[disagree_id] 
        update_outputs2 = logits2[disagree_id] 
        update_weight = weight[disagree_id] 

        loss_1, loss_2, _, _, _, _ = coteaching_loss(update_outputs, update_outputs2, update_labels, forget_rate, update_weight)
    else:
        update_labels = labels
        update_outputs = logits
        update_outputs2 = logits2

        if weight == None:
            cross_entropy_1 = F.cross_entropy(update_outputs, update_labels, reduction = 'none') 
            cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels, reduction = 'none')
        else:
            cross_entropy_1 = F.cross_entropy(update_outputs, update_labels, reduction = 'none') * weight
            cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels, reduction = 'none')* weight

        loss_1 = t.sum(update_step*cross_entropy_1)/labels.size()[0]
        loss_2 = t.sum(update_step*cross_entropy_2)/labels.size()[0]
        
    return loss_1, loss_2, ind_disagree
