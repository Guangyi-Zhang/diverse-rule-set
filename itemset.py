import numpy as np
import pandas as pd


class Itemset(object):
    '''
    We use frozenset to represent an itemset.
    '''
    items: list = None
    dbs: dict = None
    labels: list = None

    @staticmethod
    def set_items(items: list):
        Itemset.items = items

    @staticmethod
    def clear_db():
        Itemset.dbs = dict()
        Itemset.labels = list()

    @staticmethod
    def set_db(l: int, db: list):
        Itemset.labels.append(l)
        Itemset.labels = sorted(Itemset.labels)
        Itemset.dbs[l] = db

    @staticmethod
    def db2idx(sign):
        base = sum([len(Itemset.dbs[l]) for l in Itemset.labels if sign>l])
        return [j+base for j, t in enumerate(Itemset.dbs[sign])]

    def __init__(self, s: set, supp=False):
        if len(s) > len(Itemset.items):
            raise Exception('num of items < {}'.format(len(s)))
        self.s = frozenset(s)
        if supp is True:
            self._cov = dict()
            for l in Itemset.labels:
                self._cov[l] = set([j for j,t in zip(Itemset.db2idx(l), Itemset.dbs[l]) if self.cover(t)])

    def __len__(self) -> int:
        return len(self.s)

    def support(self, signs: list) -> int:
        signs = set(signs)
        return sum([self._support(l) for l in Itemset.labels if l in signs])

    def _support(self, sign: int) -> int:
        return len(self._cov[sign])

    def coverage(self, signs: list) -> set:
        '''Return a set of trans ids'''
        signs = set(signs)
        return set.union(*[self._cov[l] for l in Itemset.labels if l in signs])

    def cover(self, T) -> bool:
        if len(self) > len(T):
            return False

        diff = self.s.difference(T.s)
        return True if len(diff)==0 else False

    def itemdiff(self, other):
        return self.s.difference(other.s)


class Rule(Itemset):
    '''
    Rules that contain the same set of items are considered the same.
    '''
    @staticmethod
    def quality(S: list, metric: str='kl') -> float:
        '''A modular quality'''
        if metric == 'kl':
            return sum([s.kl for s in S])
        if metric == 'acc':
            return sum([s.acc for s in S])
        raise Exception('')

    def __init__(self, s: set, l: int):
        super().__init__(s, supp=True)
        self.label = l

    def __eq__(self, other) -> bool:
        '''
        frozenset is hashable.
        '''
        return self.s == other.s and self.label == other.label

    def __hash__(self):
        return hash(self.s) ^ hash(self.label)

    @property
    def kl(self) -> int:
        '''KL distance'''
        supps = np.array([self.support([l_]) for l_ in Itemset.labels])
        if sum(supps) == 0:
            return 0

        ns = np.array([len(Itemset.dbs[l_]) for l_ in Itemset.labels])
        p = supps/sum(supps)
        p = np.where(p > 1e-9, p, 1e-9)
        q = ns/sum(ns) # q wouldn't be zero
        kl = np.sum(np.where(p != 0, p * np.log(p / q), 0))

        supp_l = self.support([self.label])
        imb_l = self.support([self.label])/sum(supps) - len(Itemset.dbs[self.label])/sum(ns)
        supp_l = supp_l if imb_l > 0 else 0
        return np.sqrt(supp_l) * kl

    @property
    def acc(self) -> float:
        '''TP / (TP + FP)'''
        dnm = self.support(Itemset.labels)
        if dnm == 0:
            return 0.0
        return self.support([self.label]) / dnm

    def overlap(self, S: list, card=False) -> float:
        if card:
            c = self.coverage(Itemset.labels)
            return [set.intersection(c, s.coverage(Itemset.labels)) for s in S]
        else:
            return sum([self._overlap(s) for s in S])

    def _overlap(self, s) -> float:
        '''Jaccard distance'''
        c = self.coverage(Itemset.labels)
        cs = s.coverage(Itemset.labels)
        cap = set.intersection(c, cs)
        if len(cap) == 0:
            return 1
        cup = set.union(c, cs)
        return 1 - len(cap) / len(cup)

    def trans(self, labels=None):
        if labels is None:
            labels = Itemset.labels
        ll = [self._cov2db(l) for l in labels]
        return [em for sl in ll for em in sl]

    def _cov2db(self, label):
        cov = self.coverage([label])
        return [t for j, t in zip(Itemset.db2idx(label), Itemset.dbs[label]) if j in cov]


class Transaction(Itemset):

    def __init__(self, s: set):
        super().__init__(s, supp=False)


def prep_db(X: pd.DataFrame, y: np.ndarray):
    '''X: Each row is a transaction'''
    Itemset.set_items(range(X.shape[1]))
    Itemset.set_items(range(X.shape[1]))
    Itemset.dbs = dict()

    for l in y.unique().tolist():
        X_ = X[y == l]
        db = _prep_db(X_ if type(X_) == np.ndarray else X_.values)
        Itemset.set_db(l, db)


def _prep_db(X: np.ndarray):
    return [Transaction(feat2item(t)) for t in X]


def feat2item(x: list):
    '''Return active items'''
    return np.nonzero(x)[0]
