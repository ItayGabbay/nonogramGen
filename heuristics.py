from nonogram import Nonogram, BOARD_SIZE, NUM_ROWS, NUM_COLS
import math
from utils import copy_and_transpose_matrix

def _ones_diff_helper(matrix, clues):
    actual_sums = [r.sum() for r in matrix]
    expected_sums = [math.fsum(r) for r in clues]
    diffs = [math.fabs(actual_sums[i] - expected_sums[i]) for i in range(len(clues))]
    return (BOARD_SIZE - math.fsum(diffs)) / BOARD_SIZE


def _zeros_diff_helper(max_val, matrix, clues):
    actual_sums = [max_val - r.sum() for r in matrix]
    expected_sums = [max_val - math.fsum(r) for r in clues]
    diffs = [math.fabs(actual_sums[i] - expected_sums[i]) for i in range(len(clues))]
    return (BOARD_SIZE - math.fsum(diffs)) / BOARD_SIZE


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
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            clue = nonogram.get_row_clue(row, col)
            if clue == -1:
                raise Exception("Indexes are off in compare_blocks_rows! row: %d col: %d matrix size: (%d, %d)" %
                                (row, col, len(nonogram.matrix), len(nonogram.matrix[0])))
            diff_sum += math.fabs(clue - nonogram.matrix[row][col])
    return diff_sum

# (6)
def compare_blocks_cols(nonogram: Nonogram):
    diff_sum = 0
    for col in range(NUM_COLS):
        for row in range(NUM_ROWS):
            clue = nonogram.get_col_clue(col, row)
            if clue == -1:
                raise Exception("Indexes are off in compare_blocks_cols! row: %d col: %d matrix size: (%d, %d)" %
                                (row, col, len(nonogram.matrix), len(nonogram.matrix[0])))
            diff_sum += math.fabs(clue - nonogram.matrix[row][col])
    return diff_sum