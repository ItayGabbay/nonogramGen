import pickle
import random

import numpy as np
from config import pickle_unsolved_file_path, pickle_solved_file_path, train_size, unsolved_nonograms_archive_name
from typing import List
from config import all_rows_archive_name
from individual import DoubleTreeBasedIndividual
from klepto.archives import dir_archive


def copy_and_transpose_matrix(matrix):
    trans = np.copy(matrix)
    trans = np.transpose(trans)
    return trans


# TODO remove unnecessary funcs
def _load_pickled(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# def load_unsolved_nonograms_from_file(path: str = pickle_unsolved_file_path):
#     return _load_pickled(path)

def load_unsolved_nonograms_from_file(path: str = unsolved_nonograms_archive_name):
    archive = dir_archive(path, {}, serialized=True)
    archive.load()
    return archive['nonograms']


def load_solved_nonograms_from_file(path: str = pickle_solved_file_path):
    return _load_pickled(path)


# def load_all_row_opts(path=pickle_row_options_path):
#     return _load_pickled(path)

def load_all_row_opts(path=all_rows_archive_name):
    archive = dir_archive(path, {}, serialized=True)
    archive.load()
    return archive


def load_train_and_test_sets(solved_path=pickle_solved_file_path, unsolved_path=unsolved_nonograms_archive_name,
                             train_set_size=train_size):
    # solved_nonograms = load_solved_nonograms_from_file(solved_path)
    unsolved_nonograms = load_unsolved_nonograms_from_file(unsolved_path)

    if train_set_size > len(unsolved_nonograms):
        raise Exception("Train size is bigger than the number of available Nonograms!")

    train_indxs = random.sample(range(1, len(unsolved_nonograms)), train_set_size)
    test_indxs = filter(lambda x: x not in train_indxs, range(len(unsolved_nonograms)))
    train_nonos = [{'unsolved': unsolved_nonograms[i], 'solved': None} for i in train_indxs]
    test_nonos = [{'unsolved': unsolved_nonograms[i], 'solved': None} for i in test_indxs]
    return {'train': train_nonos, 'test': test_nonos}


def load_solved_bomb_nonogram_from_file(path: str = pickle_solved_file_path):
    return _load_pickled(path)


# def individual_to_str(individual) -> str:
#     res_lst = ['COND TREES:\n']
#     for tree in individual['CONDITION_TREES']:
#         res_lst.append(str(tree) + '\n')
#     res_lst.append('VAL TREES:\n')
#     for tree in individual['VALUE_TREES']:
#         res_lst.append(str(tree) + '\n')
#     return "".join(res_lst)

def load_unsolved_bomb_nonogram_from_file(path: str = pickle_unsolved_file_path):
    return load_unsolved_nonograms_from_file(path)[0]


def individual_to_str(individual: DoubleTreeBasedIndividual) -> str:
    res_lst = ['COND TREES:\n']
    for tree in individual.cond_trees:
        res_lst.append(str(tree) + '\n')
    res_lst.append('VAL TREES:\n')
    for tree in individual.value_trees:
        res_lst.append(str(tree) + '\n')
    return "".join(res_lst)


def individual_lst_to_str(lst: List, max_to_print=-1):
    if max_to_print < 0:
        max_to_print = len(lst)
    res_lst = [individual_to_str(i) for i in lst[:max_to_print]]
    return "".join(res_lst)
