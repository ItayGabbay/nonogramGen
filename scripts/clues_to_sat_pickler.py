from utils import load_all_row_opts, pickle_row_options_path
from config import pickle_row_options_path
import pickle
from typing import List, Callable
import numpy as np
import math

all_opts_dict = load_all_row_opts('../' + pickle_row_options_path)


def avg(lst):
    return np.sum(lst)


def clues_to_sat_single_row(clues: List[int], and_op=np.mean, or_op=np.max):
    # inner func definitions:
    def indices_wrapper(indices: np.ndarray):
        def func_for_opt(row):
            vals = []
            for i, val in enumerate(row):
                if i in indices:
                    vals.append(val)
                else:
                    vals.append(not val)
            return and_op(vals)
        return func_for_opt

    def or_between_opts_wrapper(opts_funcs: List[Callable]):
        def or_between_opts(row):
            return or_op([func(row) for func in opts_funcs])
        return or_between_opts

    # actual code:
    opts = all_opts_dict[str(clues)]
    funcs = []
    for option in opts:
        true_indxs = np.nonzero(option)[0]
        f = indices_wrapper(true_indxs)
        funcs.append(f)
    return or_between_opts_wrapper(funcs)


r = [True, False, True, False, True]
c = [1, 1]
foo = clues_to_sat_single_row(c)
res = foo(r)
print(res)