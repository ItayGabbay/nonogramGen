from typing import List
import numpy as np
from config import NUM_COLS, NUM_ROWS, empty_in_split,convert_to_sat
from nonogram_to_sat import clues_to_sat

BOARD_SIZE = NUM_ROWS * NUM_COLS


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
    def __init__(self, row_clues: str, col_clues: str, title: str, number: int, solution_url: str,
                 matrix: np.array = None) -> None:
        # metadata
        self.title = title
        self.number = number

        # actual data
        self.solution_url = solution_url
        if type(row_clues) is str:
            self.row_clues = _parse_clues_rows(row_clues)
        else:
            self.row_clues = row_clues
        if type(col_clues) is str:
            self.col_clues = _parse_clues_cols(col_clues)
        else:
            self.col_clues = col_clues
        if matrix is None:
            self.matrix = np.full((NUM_ROWS, NUM_COLS), False, dtype=bool)
        else:
            self.matrix = np.copy(matrix)

        if convert_to_sat:
            self.converter_to_sat = clues_to_sat(self.col_clues, self.row_clues)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return 'title: %s, \nnumber: %d, \nrow clues: %s,\ncol clues: %s,\nmatrix: %s \nsolution: %s' % \
               (self.title, self.number, self.row_clues, self.col_clues, self.matrix, self.solution_url)

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

    def convert_to_sat(self, and_op=np.mean):
        if convert_to_sat:
            row_res = [self.converter_to_sat(r) for r in self.matrix]
            res = and_op(row_res)
            return res
        else:
            raise Exception('convert to sat is disabled but convert_to_sat was called!')
