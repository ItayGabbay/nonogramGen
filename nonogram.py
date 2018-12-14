from typing import List
import math

import numpy as np

from utils import copy_and_transpose_matrix

NUM_ROWS = 20
# NUM_ROWS = 5
NUM_COLS = 20
# NUM_COLS = 5
MAX_RETURN_VAL_COMP_TO_EXPECTED = NUM_ROWS * NUM_COLS
empty_in_split = '\xa0'


def _parse_clues_cols(clues: str) -> List[List[int]]:
    temp = [[] for _ in range(NUM_COLS)]
    split = clues.split(',')
    idx = 0
    for clue in split:
        if clue != empty_in_split:
            temp[idx].append(int(clue))
        idx = (idx + 1) % len(temp)
    return temp


def _parse_clues_rows(clues: str) -> List[List[int]]:
    temp = [[] for _ in range(NUM_ROWS)]
    split = clues.split(',')
    clues_per_row = int(len(split) / NUM_ROWS)
    for idx in range(len(temp)):
        for i in range(clues_per_row):
            clue = split.pop(0)
            if clue != empty_in_split:
                temp[idx].append(int(clue))
    return temp


class Nonogram(object):
    def __init__(self, row_clues: str, col_clues: str) -> None:
        super().__init__()
        self.row_clues = _parse_clues_rows(row_clues)
        self.col_clues = _parse_clues_cols(col_clues)
        self.matrix = np.full((NUM_ROWS, NUM_COLS), False, dtype=bool)

    def __str__(self) -> str:
        return 'row clues: %s,\ncol clues: %s,\nmatrix: %s' % (self.row_clues, self.col_clues, self.matrix)

    def get_row_clue(self, row_num: int, index: int) -> int:
        if row_num > len(self.row_clues) or index < 0:
            return -1
        row = self.row_clues[row_num]
        if index < len(row):
            return row[index]
        return 0

    def get_col_clue(self, col_num: int, index: int) -> int:
        if col_num > len(self.col_clues) or index < 0:
            return -1
        row = self.col_clues[col_num]
        if index < len(row):
            return row[index]
        return 0

