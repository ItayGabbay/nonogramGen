from typing import List

import numpy as np

NUM_ROWS = 20
# NUM_ROWS = 5
NUM_COLS = 20
# NUM_COLS = 5
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