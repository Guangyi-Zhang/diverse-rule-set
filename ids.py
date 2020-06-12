from interpretable_decision_sets.IDS_deterministic_local import run_apriori, createrules, \
    deterministic_local_search

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


def IDS(Xtr, Ytr, lambda_array: list, freq: float=0.1):
    Xtr = Xtr.reset_index(drop=True)
    itemsets = run_apriori(Xtr, freq)
    print('freq itemsets: ', len(itemsets))
    list_of_rules = createrules(itemsets, list(set(Ytr)))

    assert len(lambda_array) == 7
    epsilon = 0.05
    soln_set, obj_val = deterministic_local_search(list_of_rules, Xtr, Ytr, lambda_array, epsilon)
    dc = Counter(Ytr).most_common(1)[0][0]
    print('IDS: ', soln_set)
    print('IDS obj: ', obj_val)
    print('IDS default class: ', dc)

    return [list_of_rules[r_idx] for r_idx in soln_set], len(itemsets), dc


def IDS_predict(model, Xt, default=0):
    mask_array = np.array([default] * Xt.shape[0])
    if len(model) == 0:
        return mask_array

    # an ordering over rules
    accs = [len(r.get_correct_cover(None,None))/len(r.get_cover(None)) for r in model]
    order = np.argsort(accs)[::-1]

    Xt = Xt.reset_index(drop=True)
    used = set()
    for i in order:
        r = model[i]
        preds = set(r.get_cover(Xt, reuse=False))
        preds = preds.difference(used)
        used.union(set(preds))
        preds = np.array(list(preds))
        mask_array[preds] =  r.class_label

    return mask_array
