import pickle
import numpy as np
from config import pickle_file_path
from typing import List
from nonogram import Nonogram


def copy_and_transpose_matrix(matrix):
    trans = np.copy(matrix)
    trans = np.transpose(trans)
    return trans

def load_nonograms_from_file(path: str = pickle_file_path) -> List[Nonogram]:
    with open(path, 'rb') as f:
        return pickle.load(f)