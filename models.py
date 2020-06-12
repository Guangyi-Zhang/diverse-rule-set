import numpy as np
import pandas as pd
from functools import partial
import itertools

import Orange
from Orange.data import Domain, DiscreteVariable

from sklearn import tree
from sklearn.metrics import precision_score, recall_score, f1_score, \
    roc_auc_score, balanced_accuracy_score, confusion_matrix
roc_auc_score = partial(roc_auc_score, multi_class='ovo', average='macro')

from decisionset import DecisionSet
from itemset import Itemset, Rule, Transaction, prep_db, feat2item
from util import recall, precision, dispersion, printRules, dispersion_, overlap
from util import max_freq_ratio, tree2rules
from ids import IDS, IDS_predict

from pyarc import CBA, TransactionDB


def run_ids(Xtr, Ytr, Xt, Yt, lb, min_freq, lambs, log=None):
    ids, nfreq, default = IDS(Xtr, Ytr.values, lambs, freq=min_freq)
    for r in ids:
        print('class: ', r.class_label, ', cover: {}/{}'.format(len(r.get_correct_cover(Xtr, Ytr)),
                                                                len(r.get_cover(Xtr))), end='; ')
        r.print_rule()

    for r in ids:
        r.covered = set(r.get_cover(Xtr))

    Y_pred = IDS_predict(ids, Xt, default=default)

    if log is None:
        from logger import log
    [log('ids-lambda', lamb, i) for i, lamb in enumerate(lambs)]
    log('ids-k', len(ids))
    [log('ids-nconds', r.get_length(), i) for i, r in enumerate(ids)]
    log('ids-nfreq', nfreq)
    log('ids-freq', min_freq)
    log('ids-default', default)
    log('ids-auc', roc_auc_score(lb.transform(Yt.values), lb.transform(Y_pred)))
    log('ids-bacc', balanced_accuracy_score(Yt, Y_pred))
    log('ids-disp', dispersion_(ids, average=True))
    log('ids-overlap', overlap(ids))
    print(confusion_matrix(Yt, Y_pred))

    return Y_pred


def run_cn2(Xtr, Ytr, Xt, Yt, lb, k=None, log=None):
    domainx = Domain.from_numpy(Xtr.values)
    domainy = Domain.from_numpy(Ytr.values.reshape((-1, 1)))
    datax = Orange.data.Table.from_numpy(domainx, Xtr.values)
    datay = Orange.data.Table.from_numpy(domainy, Ytr.values.reshape((-1, 1)))
    discretizer = Orange.preprocess.DomainDiscretizer()
    domainx = discretizer(datax)
    domainy = discretizer(datay)
    domain = Domain(domainx.attributes, domainy.attributes[0])
    data = Orange.data.Table.from_numpy(domain, Xtr.values, Y=Ytr.values)

    learner = Orange.classification.CN2UnorderedLearner()
    #learner = Orange.classification.rules.CN2Learner()
    learner.rule_finder.search_algorithm.beam_width = 10
    learner.rule_finder.search_strategy.constrain_continuous = True
    learner.rule_finder.general_validator.min_covered_examples = 15
    cn2 = learner(data)


    if k is not None:
        r_def = cn2.rule_list[-1]
        cn2.rule_list = cn2.rule_list[:k]
        cn2.rule_list.append(r_def)

    Y_pred = np.argmax(cn2.predict(Xt.values), axis=1)

    ids = np.arange(Xtr.shape[0])
    print('default:', cn2.rule_list[-1].prediction)
    # Skip the last default rule
    for i,r in enumerate(cn2.rule_list[:-1]):
        cov = np.array([r.evaluate_instance(x) for x in data])
        pred = np.array([r.prediction] * sum(cov))
        acc = pred == Ytr.values[cov]
        r.covered = set(ids[cov])
        print('CN2', '#{}, label:{}, len:{}, cov:{}, acc:{}'.format(i, r.prediction, r.length, sum(cov)/len(ids), sum(acc)/sum(cov)))

    if log is None:
        from logger import log
    log('cn2-k', len(cn2.rule_list[:-1]))
    [log('cn2-nconds', r.length, i) for i, r in enumerate(cn2.rule_list[:-1])]
    log('cn2-auc', roc_auc_score(lb.transform(Yt.values), lb.transform(Y_pred)))
    log('cn2-bacc', balanced_accuracy_score(Yt, Y_pred))
    log('cn2-disp', dispersion_(cn2.rule_list[:-1], average=True))
    log('cn2-overlap', overlap(cn2.rule_list[:-1]))
    print(confusion_matrix(Yt, Y_pred))


def run_cart(Xtr, Ytr, Xt, Yt, lb, k=None, log=None):
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None, max_leaf_nodes=k)
    clf = clf.fit(Xtr, Ytr)
    print('tree depth and #leaf: ', clf.get_depth(), clf.get_n_leaves())
    leaves = tree2rules(clf, Xtr.columns)
    Y_pred = clf.predict(Xt)

    if log is None:
        from logger import log
    log('CART-k', clf.get_n_leaves())
    [log('CART-nconds', len(r.split('AND')), i) for i, r in enumerate(leaves.split('\n'))]
    log('CART-depth', clf.get_depth())
    log('CART-rules', leaves)
    log('CART-auc', roc_auc_score(lb.transform(Yt.values), lb.transform(Y_pred)))
    log('CART-bacc', balanced_accuracy_score(Yt, Y_pred))
    log('CART-overlap', 0)
    print(confusion_matrix(Yt, Y_pred))


def run_cba(Xtr, Ytr, Xt, Yt, lb, support=0.20, confidence=0.5, k=None, log=None):
    txns_train = TransactionDB.from_DataFrame(pd.concat([Xtr, Ytr], axis=1))
    txns_test = TransactionDB.from_DataFrame(pd.concat([Xt, Yt], axis=1))
    cba = CBA(support=support, confidence=confidence, algorithm="m1")
    cba.fit(txns_train)

    if k is not None:
        cba.clf.rules = cba.clf.rules[:k]

    Y_pred = [int(i) for i in cba.predict(txns_test)]

    for r in cba.clf.rules:
        r.covered = set([i for i, rd in enumerate(txns_train) if r.antecedent <= rd])

    if log is None:
        from logger import log
    log('cba-k', len(cba.clf.rules))
    log('cba-rules', str(cba.clf.rules))
    [log('cba-nconds', len(r), i) for i, r in enumerate(cba.clf.rules)]
    log('cba-auc', roc_auc_score(lb.transform(Yt.values), lb.transform(Y_pred)))
    log('cba-bacc', balanced_accuracy_score(Yt, Y_pred))
    log('cba-disp', dispersion_(cba.clf.rules, average=True))
    log('cba-overlap', overlap(cba.clf.rules))
    print(confusion_matrix(Yt, Y_pred))


def run_ours(Xtr, Ytr, Xt, Yt, lb, nsample, lambda_mode, q, sample_mode, k=None, rerun=True,
             eps=0.01, min_recall_per_class=0.8,
             log=None):
    #name = 'ours' if k is None else 'oursk'
    name = 'ours{}'.format(int(rerun))
    k = k if k is not None else 100

    dec = DecisionSet(eps)
    dec.train(Xtr, Ytr, max_k=k, nsamp=nsample, lamb=lambda_mode, q=q, mode=sample_mode, rerun=rerun, min_recall_per_class=min_recall_per_class)
    print('default:', dec.default)

    Xt_ = [Transaction(feat2item(t)) for t in Xt.values]
    Y_pred = dec.predict_all(Xt_)

    if log is None:
        from logger import log
    log('{}-default'.format(name), dec.default)
    log('{}-k'.format(name), len(dec.rules))
    log('{}-maxk'.format(name), k)
    [log('{}-nconds'.format(name), len(r), i) for i, r in enumerate(dec.rules)]
    log('{}-q'.format(name), q)
    log('{}-nsample'.format(name), nsample)
    log('{}-lamb'.format(name), lambda_mode)
    log('{}-seq'.format(name), dec.seq)
    log('{}-auc'.format(name), roc_auc_score(lb.transform(Yt.values), lb.transform(Y_pred)))
    log('{}-bacc'.format(name), balanced_accuracy_score(Yt, Y_pred))
    log('{}-disp'.format(name), dispersion(dec.rules, average=True))
    log('{}-overlap'.format(name), overlap(dec.rules))
    log('{}-mode'.format(name), sample_mode)
    [log('{}-precisions-tr'.format(name), v, l) for l, v in precision(dec).items()]
    [log('{}-recall-tr'.format(name), v, l) for l, v in recall(dec.rules).items()]
    print(confusion_matrix(Yt, Y_pred))

    return Y_pred
