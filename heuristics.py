from nonogram import Nonogram, BOARD_SIZE, NUM_ROWS, NUM_COLS
import math
from utils import copy_and_transpose_matrix
from typing import List
import numpy as np


def _ones_diff_helper(matrix, clues):
    actual_sums = [r.sum() for r in matrix]
    expected_sums = [math.fsum(r) for r in clues]
    diffs = [math.fabs(actual_sums[i] - expected_sums[i]) for i in range(len(clues))]
    res = (BOARD_SIZE - math.fsum(diffs)) / BOARD_SIZE
    # print('_ones_diff_helper returned', res)
    return res


def _zeros_diff_helper(max_val, matrix, clues):
    actual_sums = [max_val - r.sum() for r in matrix]
    expected_sums = [max_val - math.fsum(r) for r in clues]
    diffs = [math.fabs(actual_sums[i] - expected_sums[i]) for i in range(len(clues))]
    res = (BOARD_SIZE - math.fsum(diffs)) / BOARD_SIZE
    # print('_zeros_diff_helper returned', res)
    return res


def _find_clue_for_row(row_to_check: List[bool]) -> List[int]:
    def skip_till_val(start=0, val=True):
        if start >= len(row_to_check):
            return len(row_to_check), -1

        count = 0
        while start < len(row_to_check) and row_to_check[start] != val:
            start += 1
            count += 1
        return start, count

    clues_to_ret = []
    block_idx, block_len = skip_till_val()
    while block_idx < len(row_to_check):
        block_idx, block_len = skip_till_val(block_idx, False)
        clues_to_ret.append(block_len)
        block_idx, block_len = skip_till_val(block_idx)
    return clues_to_ret


def _make_same_len(lst1, lst2):
    def padd_with_zeros(to_padd, final_len):
        while len(to_padd) < final_len:
            to_padd.append(0)

    if len(lst1) < len(lst2):
        padd_with_zeros(lst1, len(lst2))
    elif len(lst1) > len(lst2):
        padd_with_zeros(lst2, len(lst1))

    return lst1, lst2


# heuristics are taken from "Solving Nonograms Using Genetic Algorithms"

# (1)
def ones_diff_rows(nonogram: Nonogram) -> float:
    """
    determines the difference between the expected number and the current number of 1’s in matrix, per row
    """
    return _ones_diff_helper(nonogram.matrix, nonogram.row_clues)


# (2)
def ones_diff_cols(nonogram: Nonogram):
    """
    determines the difference between the expected number and the current number of 1’s in matrix, per col
    """
    trans = copy_and_transpose_matrix(nonogram.matrix)
    return _ones_diff_helper(trans, nonogram.col_clues)


# (3)
def zeros_diff_rows(nonogram: Nonogram):
    """
    the difference between the number of 0’s for rows
    """
    return _zeros_diff_helper(NUM_COLS, nonogram.matrix, nonogram.row_clues)


# (4)
def zeros_diff_cols(nonogram: Nonogram):
    """
    the difference between the number of 0’s for cols
    """
    trans = copy_and_transpose_matrix(nonogram.matrix)
    return _zeros_diff_helper(NUM_ROWS, trans, nonogram.col_clues)


# (5)
def compare_blocks_rows(nonogram: Nonogram):
    diff_sum = 0
    row_clues_sum = np.sum([np.sum(clues) for clues in nonogram.row_clues])
    for row in range(NUM_ROWS):
        row_blocks = _find_clue_for_row(nonogram.matrix[row])
        row_clues = nonogram.row_clues[row]
        row_blocks, row_clues = _make_same_len(row_blocks, row_clues)
        if len(row_blocks) == 0:
            continue

        actual = 0
        for block, clue in zip(row_blocks, row_clues):
            val = math.fabs(clue - block)
            actual += val
        diff_sum += actual
    res = (row_clues_sum - diff_sum) / row_clues_sum
    return res


# (6)
def compare_blocks_cols(nonogram: Nonogram):
    diff_sum = 0
    col_clues_sum = np.sum([np.sum(clues) for clues in nonogram.col_clues])
    matrix = copy_and_transpose_matrix(nonogram.matrix)
    for col in range(NUM_COLS):
        col_blocks = _find_clue_for_row(matrix[col])
        col_clues = nonogram.row_clues[col]
        col_blocks, col_clues = _make_same_len(col_blocks, col_clues)
        if len(col_blocks) == 0:
            continue

        actual = 0
        for block, clue in zip(col_blocks, col_clues):
            val = math.fabs(clue - block)
            actual += val
        diff_sum += actual
    res = (col_clues_sum - diff_sum) / col_clues_sum
    return res


def get_max_row_clue(nonogram: Nonogram):
    max_row_clue = 0

    for row_index in range(len(nonogram.row_clues)):
        for clue_index in range(len(nonogram.row_clues[row_index])):
            if max_row_clue < nonogram.row_clues[row_index][clue_index]:
                max_row_clue = nonogram.row_clues[row_index][clue_index]

    res = float(max_row_clue) / NUM_ROWS
    # print('get_max_row_clue returned', res)
    return res


def get_max_col_clue(nonogram: Nonogram):
    max_col_clue = 0

    for col_index in range(len(nonogram.col_clues)):
        for clue_index in range(len(nonogram.col_clues[col_index])):
            if max_col_clue < nonogram.col_clues[col_index][clue_index]:
                max_col_clue = nonogram.col_clues[col_index][clue_index]

    res = float(max_col_clue) / NUM_COLS
    # print('get_max_col_clue returned', res)
    return res
