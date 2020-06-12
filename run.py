import numpy as np
import pandas as pd
from os import path
import pickle
import itertools
from functools import partial
import pprint
pp = pprint.PrettyPrinter(indent=4)

from experiment import new_exp


def run_supervise(ver, use_min_freq, ds, k):
    nsample, lambda_mode = 500, 'max'

    ex = new_exp(interactive=False)

    data_lamb = {
        'iris': [1, 1, 1, 1, 10, 1, 1],
        'contracept': [1, 1, 1, 1, 1, 1, 100],
        'cardio': [1, 1, 1, 1, 1, 1, 100],
        'anuran': None,
        'avila': None,
    }
    data_min_freq = {
        'iris': 0.1,
        'contracept': 0.1,
        'cardio': 0.5,
        #'anuran': 0.5,
        #'avila': 0.8,
        'anuran': None,
        'avila': None,
    }

    for ver, it, nsample_s, lambda_mode_s, mode, dataset in \
            itertools.product([int(ver)],
                              #range(1,3+1),
                              range(1,1+1),
                              [nsample],
                              [lambda_mode],
                              #[0,2],
                              [2],
                              #['poker0.1'],
                              ds.split(','),
                              #['iris', 'contracept', 'cardio', 'anuran', 'avila'],
                              ):
        conf = {
            'dataset': dataset,
            'ver': ver,
            'it': it,
            'nsample': nsample_s,
            'lambda_mode': lambda_mode_s,
            'quality': 'kl',
            #'lambda_array': None,
            'lambda_array': data_lamb.get(dataset,None),
            'min_freq': data_min_freq.get(dataset,None) if use_min_freq else None,
            'sample_mode': mode,
            'k': k,
        }
        pp.pprint(conf)
        r = ex.run(config_updates=conf)


if __name__ == '__main__':
    import sys
    ver = sys.argv[1]
    use_min_freq = False if sys.argv[2] == 'False' else True
    ds = sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv)>4 else None
    use_test = None

    run_supervise(ver, use_min_freq, ds, k)
