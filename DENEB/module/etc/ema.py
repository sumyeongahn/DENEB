import io
import torch as t
import numpy as np
import torch.nn as nn
from module.loss.elr import sigmoid_rampup

class EMA:
    def __init__(self, label = None, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.parameter = t.zeros(label.size(0)).cuda()
        self.updated = t.zeros(label.size(0)).cuda()
        

    def update(self, data, index):
        self.parameter[index] = self.alpha * self.parameter[index] + (1-self.alpha*self.updated[index]) * data

        self.updated[index] = 1
        
    def max(self, label=None):
        if label == None or t.sum(self.label ==label) == 0:
            return self.parameter.max()    
        else:
            label_index = t.where(self.label == label)[0]
            return self.parameter[label_index].max()

    def min(self, label=None):
        if label == None or t.sum(self.label ==label) == 0:
            return self.parameter.min()    
        else:
            label_index = t.where(self.label == label)[0]
            return self.parameter[label_index].min()





def update_ema_variables(model, model_ema, global_step, ema_update, ema_step, alpha_=0.997):
    if alpha_ == 0:
        ema_param.data = param.data
    else:
        if ema_update:
            alpha = sigmoid_rampup(global_step + 1, ema_step ) * alpha_
        else:
            alpha = min(1 - 1 / (global_step + 1), alpha_)
        for ema_param, param in zip(model_ema.parameters(), model.parameters()):
            ema_param.data = alpha * ema_param.data + (1 - alpha) * param.data

