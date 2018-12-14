import numpy as np


def copy_and_transpose_matrix(matrix):
    trans = np.copy(matrix)
    trans = np.transpose(trans)
    return trans