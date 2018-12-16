import pickle
import numpy as np
from config import pickle_unsolved_file_path, pickle_solved_file_path
from typing import List
from nonogram import Nonogram


def copy_and_transpose_matrix(matrix):
    trans = np.copy(matrix)
    trans = np.transpose(trans)
    return trans

def load_unsolved_nonograms_from_file(path: str = pickle_unsolved_file_path) -> List[Nonogram]:
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_solved_nonograms_from_file(path: str = pickle_solved_file_path) -> List[Nonogram]:
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_solved_bomb_nonogram_from_file(path: str = pickle_solved_file_path) -> Nonogram:
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_unsolved_bomb_nonogram_from_file(path: str = pickle_unsolved_file_path) -> Nonogram:
    return load_unsolved_nonograms_from_file(path)[0]
