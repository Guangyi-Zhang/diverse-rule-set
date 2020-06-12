from itemset import Itemset, Rule, Transaction

import numpy as np
import collections
from itertools import combinations


def tree2rules(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, conds):
        if (threshold[node] != -2):
            cond = '( ' + features[node] + ' {} ' + str(threshold[node]) + ' )'
            cond = 'IF ' + cond if len(conds)==0 else ' AND ' + cond
            if left[node] != -1:
                leaves_l = recurse (left, right, threshold, features, left[node], conds + cond.format('<='))
            if right[node] != -1:
                leaves_r = recurse (left, right, threshold, features,right[node], conds + cond.format('>'))
            return '\n'.join([leaves_l,leaves_r])
        else:
            print(conds, end=', ')
            print('return ' + 'data=' + str(value[node]) + ', class={}'.format(np.argmax(value[node])))
            return conds + ', ' + 'return ' + 'data=' + str(value[node]) + ', class={}'.format(np.argmax(value[node]))

    return recurse(left, right, threshold, features, 0, '')


def max_freq_ratio(Y):
    counter = collections.Counter(Y.values)
    ls = counter.most_common()
    most_common, num_most_common = ls[0]
    least_common, num_least_common = ls[-1]
    return num_most_common / num_least_common


def overlap(R):
    n = len(R)
    if n == 0:
        return -1
    if n == 1:
        return 0
    if isinstance(R[0], Rule):
        return len(set.union(*[r1.overlap([r2], card=True)[0] for r1,r2 in combinations(R,2)]))
    else:
        return len(set.union(*[r1.covered.intersection(r2.covered) for r1,r2 in combinations(R,2)]))


def dispersion_(R, average=False):
    n = len(R)
    if n == 0:
        return -1
    if n == 1:
        return 1
    disp = sum([_dispersion(r1,r2) for r1 in R for r2 in R])
    return disp/(n*(n-1)) if average else disp


def _dispersion(r1, r2):
    nm = len(r1.covered.intersection(r2.covered))
    dnm = len(r1.covered.union(r2.covered))
    return 1 - nm/dnm


def dispersion(rules, average=False) -> float:
    '''Sum of full dispersion matrix'''
    if len(rules) == 1:
        return 1
    if len(rules) == 0:
        return -1
    dnm = len(rules) * (len(rules) - 1)
    disp = sum([r.overlap(rules) for r in rules])
    if not average:
        return disp
    else:
        return disp / dnm


def precision(dec) -> float:
    prec = dict()
    for l in Itemset.labels:
        db = Itemset.dbs[l]
        pred = dec.predict_all(db)
        prec[l] = sum(pred==l) / len(db)
    return prec


def recall(rules, sep=True) -> float:
    '''This is not a real recall'''
    labels = Itemset.labels
    ret = dict()
    for l in labels:
        total = len(Itemset.dbs[l])
        if sep:
            rules_l = [r for r in rules if r.label == l]
        else:
            rules_l = rules
        recall_ = len(set([ti for r in rules_l for ti in r.coverage([l])]))
        ret[l] = recall_ / total
    return ret


def printRules(rules: list, items: list=None):
    for r in rules:
        if items is None:
            print('- (', r.label, ':', np.sort([i for i in r.s]), ')', end='; ')
        else:
            print('- (', r.label, ':', [items[i] for i in np.sort(list(r.s))], ')', end='; ')
        print('acc: {0: .3f}'.format(r.acc), end='; ')
        print('recall: {}'.format(recall([r]).items()))


def obj(rules, lamb, qtype='kl', sep=False) -> float:
    q = Rule.quality(rules, metric=qtype)
    d = dispersion(rules)
    if sep:
        return q, lamb*d
    return q + lamb * d
