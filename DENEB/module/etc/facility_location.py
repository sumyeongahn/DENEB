import numpy as np
import torch as t


class FacilityLocation:
    def __init__(self, V, D=None, fnpy=None):
        self.D = D
        self.D *= -1
        self.D -= self.D.min()
        self.V = V
        self.curVal = 0
        self.gains = []
        self.curr_max = t.zeros_like(self.D[0])

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            new_dists = t.stack([self.curr_max, self.D[ndx]], dim=0)
            return new_dists.max(dim=0)[0].sum()
        else:
            return self.D[sset + [ndx]].sum()

    def add(self, sset, ndx, delta):
        self.curVal += delta
        self.gains += delta,
        self.curr_max = t.stack([self.curr_max, self.D[ndx]], dim=0).max(dim=0)[0]
        return self.curVal
