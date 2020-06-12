import numpy as np
import pandas as pd
from functools import partial
import itertools
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from decisionset import DecisionSet
from util import max_freq_ratio
from models import run_cart,run_cba,run_cn2,run_ids,run_ours,roc_auc_score

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

from incense import ExperimentLoader

from mongodburi import mongo_uri, db_name


def get_loader(uri=mongo_uri, db=db_name):
    loader = ExperimentLoader(
        mongo_uri=uri,
        db_name=db
    )
    return loader


def load_data(fn, rn=None, log=None, onehot=False):
    fn = fn.replace('.pkl','-1hot.pkl') if onehot else fn
    with open(fn, 'rb') as f:
        X, Y = pickle.load(f)
        Xtr_, Xt, Ytr_, Yt = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=rn)
        Xtr, Xv, Ytr, Yv = train_test_split(Xtr_, Ytr_, test_size=0.2, stratify=Ytr_, random_state=rn)

        if log is None:
            from logger import log
        log('nitem{}'.format('-1hot' if onehot else ''), X.shape[1])
        log('n{}'.format('-1hot' if onehot else ''), X.shape[0])
        log('ncls{}'.format('-1hot' if onehot else ''), len(Y.unique()))
        log('imb{}'.format('-1hot' if onehot else ''), max_freq_ratio(Y))
    lb = LabelBinarizer()
    lb.fit(Y)
    return Xtr_, Xt, Ytr_, Yt, Xtr, Xv, Ytr, Yv, lb


def new_exp(uri=mongo_uri, db=db_name, interactive=True):
    ex = Experiment('jupyter_ex', interactive=interactive)
    ex.captured_out_filter = apply_backspaces_and_linefeeds
    if uri is not None and db is not None:
        ex.observers.append(MongoObserver(url=uri, db_name=db))

    @ex.config
    def my_config():
        lambda_range = [1,10,100,500,1000]
        lambda_array = None
        min_freq = 0.1
        k = 100
        freq_cba = 0.1
        conf_cba = 0.3

    @ex.main
    def my_main(_run, ver, it, dataset, k,
                nsample, lambda_mode, quality, sample_mode,
                freq_cba, conf_cba,
                min_freq, lambda_range, lambda_array):
        from logger import log
        if uri is not None and db is not None:
            log = _run.log_scalar

        fn = 'dataset/{}-bin5.pkl'.format(dataset)
        rn = np.random.randint(10000)
        #rn = 25
        Xtr_, Xt, Ytr_, Yt, Xtr, Xv, Ytr, Yv, lb = load_data(fn, rn=rn, log=log)

        Xtr__dds, Xt_dds, Ytr__dds, Yt_dds, Xtr_dds, Xv_dds, Ytr_dds, Yv_dds, lb_dds = load_data(fn, rn=rn, log=log, onehot=True)

        k = 100 if k is None else k

        # CART
        run_cart(Xtr_, Ytr_, Xt, Yt, lb, k=k, log=log)
        print()

        # CBA
        run_cba(Xtr_, Ytr_, Xt, Yt, lb, support=freq_cba, confidence=conf_cba, k=k, log=log)
        print()

        # CN2
        run_cn2(Xtr_, Ytr_, Xt, Yt, lb, k=k, log=log)

        # IDS
        if min_freq is not None:
            if lambda_array is None:
                best_la = []
                for ith in range(7):
                    best_sc = -1
                    best_lamb = None
                    for lamb in lambda_range:
                        print('=== tuning IDS: ', ith, lamb)
                        la = best_la + [0.5] * (7 - len(best_la))
                        la[ith] = lamb

                        Y_pred = run_ids(Xtr, Ytr, Xv, Yv, lb, min_freq, la, log=None)
                        auc = roc_auc_score(lb.transform(Yv.values), lb.transform(Y_pred))
                        if auc > best_sc:
                            best_sc = auc
                            best_lamb = lamb

                    best_la.append(best_lamb)

                print('best lambs: ', best_la)
                [log('ids-lambda', lamb, i) for i, lamb in enumerate(best_la)]
            else:
                best_la = lambda_array

            run_ids(Xtr_, Ytr_, Xt, Yt, lb, min_freq, best_la, log=log)
        print()

        # DDS
        run_ours(Xtr__dds, Ytr__dds, Xt_dds, Yt_dds, lb_dds, nsample, lambda_mode, q=quality, sample_mode=sample_mode, k=k, log=log)
        run_ours(Xtr__dds, Ytr__dds, Xt_dds, Yt_dds, lb_dds, nsample, lambda_mode, q=quality, sample_mode=sample_mode, k=k, rerun=False, log=log)

    return ex
