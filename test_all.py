from itemset import Itemset, Rule, Transaction, prep_db, feat2item
from div import GreedyDiv
from sampling import sample
from decisionset import DecisionSet
from util import recall, precision, dispersion

from sklearn.model_selection import train_test_split
import pickle
import pytest


def test_itemset():
    Itemset.set_items(range(5))
    Itemset.clear_db()
    ls = [0,1]
    Itemset.set_db(ls[0], [])
    Itemset.set_db(ls[1], [Transaction({1,2,3})])

    # test eq and hash
    r = Rule({1,2,3,4}, 1)
    assert len(r) == 4
    assert r == Rule({1,2,3,4}, 1)
    assert r != Rule({1,2,3,4}, 0)
    rules = [Rule({1,2,3,4}, 1), Rule({1,2,3,4}, 1)]
    l = list(set(rules))
    assert len(l) == 1
    assert l[0] == Rule({1,2,3,4}, 1)
    rules = [Rule({1,2,3,4}, 0), Rule({1,2,3,4}, 1)]
    l = list(set(rules))
    assert len(l) == 2

    assert r.support(ls) == 0
    assert Rule({4}, 1).support(ls) == 0
    assert Rule({2}, 1).support(ls) == 1

    assert r.overlap([Rule({1,2}, 1)]) - 1 < 1e-8
    assert r.overlap([Rule({1,2}, 0)]) - 1 < 1e-8
    assert Rule({1,2,3}, 1).overlap([Rule({1,2}, 1)]) - 0 < 1e-8
    
    assert r.itemdiff(Rule({1,2}, 1)) == frozenset([3,4])

    Itemset.set_db(0, [Transaction({0}),
                       Transaction({0}),
                       ])
    Itemset.set_db(1, [Transaction({0}),
                       Transaction({0}),
                       ])
    r = Rule({0}, 1)
    assert len(r.coverage(ls)) == 4
    assert len(r.coverage([0])) == 2
    assert len(r.coverage([1])) == 2

    with pytest.raises(Exception):
        Rule({1, 2, 3, 4, 5, 6}, 1)


def test_transaction():
    # Trans has nothing to do with labels and dbs.
    Itemset.set_items(range(5))
    assert Transaction({0}) != Transaction({0})

    fn = 'dataset/iris-bin5.pkl'
    with open(fn, 'rb') as f:
        X, Y = pickle.load(f)
        train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.25)
        prep_db(train_data, train_label)

        for t,y in zip(test_data.values, test_label.values):
            t = Transaction(feat2item(t))


def test_div():
    # Fake dataset
    Itemset.set_items(range(5))
    Itemset.clear_db()
    Itemset.set_db(0, [Transaction([0])])
    Itemset.set_db(1, [Transaction([1,2,3]),
                       Transaction([0,1,2]),
                       Transaction([1,2]),
                       Transaction([2,3,4]),
                      ])

    U = []
    U.append(Rule({0}, 1))
    U.append(Rule({2,3}, 1))
    U.append(Rule({2,3,4}, 1))
    U.append(Rule({1,2,3}, 1))
    U.append(Rule({1,2}, 1))

    greed = GreedyDiv(U, 10)
    rules = greed.greedy(2)
    print(rules[0].s)
    print(rules[1].s)
    assert rules[0] == Rule({1,2}, 1)
    assert rules[1] == Rule({2,3,4}, 1)

    U = []
    U.append(Rule({0}, 1))
    U.append(Rule({1,2}, 1))
    greed.update_univ(U)
    assert len(greed.U) == 1
    assert greed.U[0] == Rule({0}, 1)


def test_sampling():
    Itemset.set_items(range(4))
    Itemset.clear_db()
    db0 = set([Transaction([0]),
           ])
    db1 = set([Transaction([1,2]),
               Transaction([1,3]),
               Transaction([1]),
           ])
    Itemset.set_db(0, db0)
    Itemset.set_db(1, db1)

    n = 3000
    nsamp, samp = sample(n, [db0], db1, {}, mode=3)
    print(nsamp)
    for s,cnt in samp.items():
        if frozenset([1]) == s:
            assert (3/7-0.05)*nsamp < cnt < (3/7+0.05)*nsamp
        if frozenset([1,2]) == s:
            assert (1/7-0.05)*nsamp < cnt < (1/7+0.05)*nsamp
        if frozenset([1,3]) == s:
            assert (1/7-0.05)*nsamp < cnt < (1/7+0.05)*nsamp
        if frozenset([2]) == s:
            assert (1/7-0.05)*nsamp < cnt < (1/7+0.05)*nsamp
        if frozenset([3]) == s:
            assert (1/7-0.05)*nsamp < cnt < (1/7+0.05)*nsamp

    sel = [Rule([1,3], 1)]
    covered = set([t for r in sel for t in r.trans()])
    nsamp, samp = sample(n, [db0], db1, covered, mode=3)
    print(nsamp)
    for s,cnt in samp.items():
        if frozenset([1,2]) == s:
            assert (0.5-0.1)*nsamp < cnt < (0.5+0.1)*nsamp
        if frozenset([2]) == s:
            assert (0.5-0.1)*nsamp < cnt < (0.5+0.1)*nsamp

    nsamp, samp = sample(n, [db0], db1, covered, mode=2)
    print(nsamp)
    for s,cnt in samp.items():
        if frozenset([1]) == s:
            assert (1/3-0.1)*nsamp < cnt < (1/3+0.1)*nsamp
        if frozenset([1,2]) == s:
            assert (1/3-0.1)*nsamp < cnt < (1/3+0.1)*nsamp
        if frozenset([2]) == s:
            assert (1/3-0.1)*nsamp < cnt < (1/3+0.1)*nsamp

    sel = [Rule([1], 1), Rule([1,3], 1)]
    covered = [t for r in sel for t in r.trans()]
    assert len(covered) == 4
    covered = set([t for r in sel for t in r.trans()])
    assert len(covered) == 3


def test_decset():
    Itemset.set_items(range(5))
    Itemset.set_db(0, [Transaction([0])])
    Itemset.set_db(1, [Transaction([1,2,3]),
                       Transaction([0,1,2]),
                       Transaction([1,2]),
                       Transaction([2,3,4]),
                      ])

    dec = DecisionSet()
    dec.set_default(0)

    sel = [Rule([1,2], 1), Rule([3], 1)]
    for r in sel: dec.add(r)
    dec.build()

    assert dispersion(dec.rules) - (1-1/4) * 2 < 1e-8

    assert dec.predict(Transaction([1,2,3])) == True
    assert dec.predict(Transaction([1,2])) == True
    assert dec.predict(Transaction([2,3,4])) == True
    assert dec.predict(Transaction([4])) == False
    assert dec.predict(Transaction([0])) == False

    assert dec.predict_and_rule(Transaction([1,2,3])) == (True, Rule([1,2], 1))
