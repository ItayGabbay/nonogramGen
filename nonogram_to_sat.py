from typing import List, Callable
from utils import load_all_row_opts
import numpy as np


class OptCallable(object):
    def __init__(self, clues: List[int], all_opts_dict, and_op, indices: np.ndarray):
        self.clues = clues
        self.all_opt_dict = all_opts_dict
        self.and_op = and_op
        self.indices = indices

    # def __call__(self, indices: np.ndarray):
    #     return OptCallable(indices, self.clues, self.all_opt_dict, self.and_op, self.or_op)
    def __call__(self, row):
        vals = []
        for i, val in enumerate(row):
            if i in self.indices:
                vals.append(val)
            else:
                vals.append(not val)
        return self.and_op(vals)


# class OptCallable(object):
#     def __init__(self, indices: np.ndarray, clues: List[int], all_opts_dict, and_op, or_op):
#         self.indices = indices
#         self.clues = clues
#         self.all_opt_dict = all_opts_dict
#         self.and_op = and_op
#         self.or_op = or_op
#
#     def __call__(self, row):
#         vals = []
#         for i, val in enumerate(row):
#             if i in self.indices:
#                 vals.append(val)
#             else:
#                 vals.append(not val)
#         return self.and_op(vals)


class OrBetweenOpts(object):
    def __init__(self, opt_callables: List[OptCallable], or_op):
        self.opt_callables = opt_callables
        self.or_op = or_op

    def __call__(self, row):
        return self.or_op([call(row) for call in self.opt_callables])


class AndBetweenRows(object):
    def __init__(self, or_between_opts: List[OrBetweenOpts], and_op):
        self.or_between_opts = or_between_opts
        self.and_op = and_op

    def __call__(self, row):
        res = self.and_op([call(row) for call in self.or_between_opts])
        return res


def _clues_to_sat_single_row(clues: List[int], all_opts_dict, and_op, or_op):
    opts = all_opts_dict[str(clues)]
    funcs = []
    for option in opts:
        true_indxs = np.nonzero(option)[0]
        row_callable = OptCallable(clues, all_opts_dict, and_op, true_indxs)
        funcs.append(row_callable)
    return OrBetweenOpts(funcs, or_op)


def clues_to_sat(col_clues, row_clues, and_op=np.mean, or_op=np.max, all_opts_dict=load_all_row_opts()):
    row_cnfs = [_clues_to_sat_single_row(c, all_opts_dict, and_op, or_op) for c in row_clues]
    return AndBetweenRows(row_cnfs, and_op)

    # def _indices_wrapper(indices: np.ndarray):
    #     def _func_for_opt(row):
    #         vals = []
    #         for i, val in enumerate(row):
    #             if i in indices:
    #                 vals.append(val)
    #             else:
    #                 vals.append(not val)
    #         return and_op(vals)
    #
    #     return _func_for_opt
    #
    # def _or_between_opts_wrapper(opts_funcs: List[Callable]):
    #     def _or_between_opts(row):
    #         return or_op([func(row) for func in opts_funcs])
    #
    #     return _or_between_opts
    #
    # # actual code:
    # opts = all_opts_dict[str(clues)]
    # funcs = []
    # for option in opts:
    #     true_indxs = np.nonzero(option)[0]
    #     f = _indices_wrapper(true_indxs)
    #     funcs.append(f)
    # return _or_between_opts_wrapper(funcs)


# # TODO cols are ignored for now
# def clues_to_sat(col_clues, row_clues, and_op=np.mean, or_op=np.max,
#                   all_opts_dict=load_all_row_opts()):
#     def and_between_cnfs_wrapper(cnfs):
#         def and_between_cnfs(row):
#             return or_op([func(row) for func in cnfs])
#
#         return and_between_cnfs
#
#     row_cnfs = [_clues_to_sat_single_row(c, all_opts_dict, and_op, or_op) for c in row_clues]
#     # col_cnfs = [_clues_to_sat_single_row(c, all_opts_dict, and_op, or_op) for c in col_clues]
#     return and_between_cnfs_wrapper(row_cnfs)
