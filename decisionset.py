from itemset import Itemset, Rule, Transaction, prep_db
from div import GreedyDiv
from sampling import sample, sample_rn
from util import recall, precision, dispersion, printRules, obj

from sklearn import tree
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
from typing import Tuple
import operator


class DecisionSet(object):

    def __init__(self, eps=0.01):
        self.rules = []
        self.labels = set()
        self.recall_eps = eps
        self.seq = True # True if use sol from 1st, or o.w. 2nd round
        self.default = None

    def set_default(self, label=None):
        '''The most under-represented class'''
        if label is not None:
            self.default = label
            return label
        rc = recall(self.rules)
        idx = np.argmin([rc[label] for label in Itemset.labels])
        deft = Itemset.labels[idx]
        self.default = deft
        return self.default

    def train(self, X, Y, max_k=100, nsamp=100, lamb=None, q='kl', mode=3, rerun=True,
              min_recall_per_class=0.5):
        print('##### START #####')
        Itemset.clear_db()
        prep_db(X, Y)

        # Allow specify lamb to a certain number by users
        if type(lamb) == str or lamb is None:
            samp = self.sample_from_each_label(set(Itemset.labels), 100, set(), mode)
            if lamb == 'max':
                lamb = np.max([Rule.quality([r], metric=q) for r in samp])
            elif lamb == 'mean':
                lamb = np.mean([Rule.quality([r], metric=q) for r in samp])
            else:
                lamb = 0
            print('lamb:', lamb)

        greed = GreedyDiv([], lamb)
        U_all = []
        labels_samp = set(Itemset.labels)
        while len(self) < max_k and len(labels_samp) > 0:
            if mode == 0:
                samps = []
                for label in labels_samp:
                    _, samp = sample_rn(nsamp, label)
                    samp = [Rule(s, label) for s in list(samp)]  # Very time-consuming
                    samps.extend(samp)
                U = set(samps)
            else:
                covered = set([t for r in self.rules for t in r.trans()])
                U = self.sample_from_each_label(labels_samp, nsamp, covered, mode)
            print('nsamp (after):', len(U))
            if len(U) == 0:
                break
            U_all.extend(U)

            # Greedy
            greed.update_univ(U)
            r = greed.greedy_once()
            # Termination criteria. Also check zero sampling above.
            if self.enough(r):
                # Include at least one rule per class, except default class.
                labels_samp.remove(r.label)
                print('remove label:', r.label)
            else:
                # Print quality vs. dispersion
                q, d = obj(self.rules, lamb, sep=True)
                qr, dr = obj(self.rules + [r], lamb, sep=True)
                print('inc q vs. d: {}, {}'.format(qr-q, dr-d))

                self.add(r)
                if np.abs(recall(self.rules)[r.label] - 1.0) < 1e-8:
                    labels_samp.remove(r.label)
                print('#{} '.format(len(self.rules)), end='')
                printRules([r])

        # Consecutive greedy over all sampels
        if rerun:
            greed.clear()
            greed.update_univ(list(set(U_all)))
            rules = greed.greedy(len(self.rules))
            if obj(rules, lamb) > obj(self.rules, lamb):
                print('Full greedy wins: {} > {}'.format(obj(rules, lamb), obj(self.rules, lamb)))
                self.reset(rules)

        default = self.set_default()
        print('default:', default)

        self.build()

        print('precision: ', precision(self).items())
        print('recall (coverage): ', recall(self.rules).items())
        print('ave disp: ', dispersion(self.rules, average=True))
        print('##### END #####')

    def sample_from_each_label(self, labels_samp, nsamp, covered, mode):
        # Sample rules from each label
        samps = []
        for label in labels_samp:
            db0 = [db for l, db in Itemset.dbs.items() if l != label]
            nsamp_, samp = sample(nsamp, db0, Itemset.dbs[label], covered, mode=mode)
            if nsamp_ == 0:
                continue
            samp = [Rule(s, label) for s in list(samp)]  # Very time-consuming
            samps = samps + samp
        return set(samps)

    def add(self, rule: Rule):
        self.rules.append(rule)
        self.labels.add(rule.label)

    def reset(self, rules: list):
        self.seq = False
        self.rules = rules
        self.labels = {r.label for r in rules}

    def enough(self, r: Rule) -> bool:
        rc_cur = recall(self.rules)
        rc_aft = recall(self.rules + [r])
        if rc_aft[r.label] - rc_cur[r.label] <= self.recall_eps:
            return True
        return False

    ##################
    # For prediction
    ##################
    def build(self):
        '''Run after adding all rules'''
        self.rules = self._sort_rules(self.rules)

    def _sort_rules(self, rules):
        #vals = np.array([Rule.quality([r]) for r in rules])
        vals = np.array([Rule.quality([r], metric='acc') for r in rules])
        idx = np.argsort(-vals)
        return [rules[i] for i in idx]

    def predict_all(self, X: list) -> list:
        return np.array([self.predict(x) for x in X])

    def predict(self, x: Transaction) -> bool:
        for r in self.rules:
            if r.cover(x):
                return r.label
        return self.default

    def predict_and_rule(self, x: Transaction) -> Tuple[bool, Rule]:
        '''Return the rule with the highest discrim()'''
        for r in self.rules:
            if r.cover(x):
                return r.label, r
        return self.default, None

    ##################
    # Util
    ##################
    def __len__(self):
        return len(self.rules)
