from itemset import Itemset, Rule
from util import dispersion
from collections import Counter

import numpy as np


class Diversity(object):

    def __init__(self, U: list, lamb: float, k: int):
        self.U = U
        self.n = len(U)
        self.k = k
        self.sel = []
        self.lamb = lamb


class GreedyDiv(Diversity):
    '''
    We allow the universal set to vary across different iterations.
    k is not mandatory.
    '''

    def __init__(self, U: list, lamb: float):
        super().__init__(U, lamb, 0)

    def greedy_once(self) -> Itemset:
        '''
        Running this func changes the state of the current instance, i.e., its selection.
        '''
        if len(self.sel) == 0:
            j = np.argmax([Rule.quality([u]) for u in self.U])
            s = self.U[j]
            del self.U[j]
            self.sel.append(s)
            return s

        qs = [Rule.quality([u]+self.sel) for u in self.U]
        ds = [u.overlap(self.sel) for u in self.U]
        vals = 0.5 * np.array(qs) + self.lamb * np.array(ds)
        j = np.argmax(vals)
        s = self.U[j]
        del self.U[j]
        self.sel.append(s)
        return s

    def clear(self):
        self.sel = []

    def update_univ(self, U: list, dup=False):
        if not dup:
            # Remove dups of sel in U
            dups = [any([u == s for s in self.sel]) for u in U]
            idc = np.where(np.array(dups))[0]
            idc = set(idc.tolist())
            U = [u for i,u in enumerate(U) if i not in idc]
        # Update
        self.U = U

    def greedy(self, k: int) -> list:
        '''Consecutive runs of greedy_once without updating U.'''
        return [self.greedy_once() for i in range(k)]
