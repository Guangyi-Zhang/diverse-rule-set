from itemset import Transaction, Itemset

from collections import Counter
import numpy as np
from typing import Tuple
import itertools


def sample(n: int, db0: list, db1: set, dbc: set, mode=3, max_sz=500) -> Tuple[int, Counter]:
    '''
    dbc: covered
    db0: a list of other datsets
    return: num of samples, Counter([frozenset of items])
    '''
    db0 = set.union(*[set(db) for db in db0])
    db1, db0, dbc = set(db1), set(db0), set(dbc)
    db1 = list(db1.difference(dbc))
    if mode == 3:
        # 3 partitions
        db0 = list(db0.difference(dbc))
        dbc = list(dbc)
    if mode == 2:
        # 2 partitions
        db0 = list(db0.union(dbc))
        dbc = []

    # limit max size
    if len(db0) > max_sz:
        db0 = [db0[i] for i in np.random.choice(len(db0), max_sz)]
    if len(db1) > max_sz:
        db1 = [db1[i] for i in np.random.choice(len(db1), max_sz)]
    if len(dbc) > max_sz:
        dbc = [dbc[i] for i in np.random.choice(len(dbc), max_sz)]

    l, bins, tuples = indexing(db0, db1, dbc)
    if l == 0:
        return 0, None
    idx = sample_records(n, l, bins)

    samp = []
    for i in idx:
        j0, j1, jc = tuples[i]
        t0 = db0[j0] if len(db0)>0 else Transaction([])
        t1 = db1[j1]
        tc = dbc[jc] if len(dbc)>0 else Transaction([])

        s1 = t1.s.difference(t0.s).difference(tc.s)
        assert len(s1) > 0
        it1 = sample_items(s1, allow_empty=False)
        s2 = t1.s.difference(s1)
        it2 = sample_items(s2, allow_empty=True)
        samp.append(frozenset(it1 + it2))

    return len(samp), Counter(samp)


def sample_rn(n: int, label) -> Tuple[int, Counter]:
    db = Itemset.dbs[label]
    l = len(db)
    samp = []
    u = np.random.randint(l, size=n)
    for i in u:
        it = sample_items(db[i].s, allow_empty=True)
        samp.append(frozenset(it))

    return len(samp), Counter(samp)


def sample_items(items: list, allow_empty=True) -> list:
    l = len(items)
    r = np.random.randint(2, size=l)
    while not allow_empty:
        if any(r): break
        r = np.random.randint(2, size=l)

    return [i for j,i in enumerate(items) if r[j]==1]


def sample_records(n: int, l: int, bins: list) -> list:
    '''Return indices of sampled records in db'''
    u = np.random.randint(l, size=n)
    return np.digitize(u, bins, right=False) # [a,b)


def indexing(db0: list, db1: list, dbc: list) -> Tuple[int, list, list]:
    if len(dbc) == 0:
        dbc = [Transaction([])]
    if len(db0) == 0:
        db0 = [Transaction([])]
    if len(db1) == 0:
        return 0, [], []

    bins = []
    tuples = []
    cur = 0
    for j0, j1, jc in itertools.product(range(len(db0)),
                                        range(len(db1)),
                                        range(len(dbc))):
        t0, t1, tc = db0[j0], db1[j1], dbc[jc]
        s1 = t1.s.difference(t0.s).difference(tc.s)
        s2 = t1.s.difference(s1)
        l = (2 ** len(s1) - 1) * 2 ** len(s2)
        if l == 0:
            continue
        cur = cur + l
        bins.append(cur)
        tuples.append((j0,j1,jc))

    return cur, np.array(bins), tuples
