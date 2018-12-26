import copy
from utils import load_all_row_opts
from typing import List


def evaluate_individual(individual, step):
    next_steps = generate_next_steps(step)

    # while len(next_steps) > 0:
    #     print "AAA"


def generate_next_steps(current_step):
    next_steps = []
    clues = dict()
    clues['rows'] = current_step.row_clues
    clues['cols'] = current_step.col_clues

    for row in range(len(current_step.matrix)):
        for col in range(len(current_step.matrix[0])):
            candidate = current_step.matrix.copy()

            if candidate[row][col] == False:
                candidate[row][col] = True
                if validate_board(candidate, clues) is True:
                    new_nonogram = copy.deepcopy(current_step)
                    new_nonogram.matrix = candidate
                    next_steps.append(new_nonogram)

    return next_steps


# def _generate_clues_combinations(clues_list):
#     combs = []
#
#
#     return combs
#
#
# def generate_next_steps_rows(current_step):
#     next_steps = []
#     clues = dict()
#     clues['rows'] = current_step.row_clues
#     clues['cols'] = current_step.col_clues
#
#     for row in range(len(current_step.matrix)):
#         combinations = _generate_clues_combinations(clues['rows'][row])

all_options = load_all_row_opts()


def is_option_possible(option: List, curr_row: List):
    for actual_cell, option_cell in zip(curr_row, option):
        if actual_cell and not option_cell:
            return False
    return True


def generate_next_steps_blocks_options(current_step):
    next_steps = []
    # clues = dict()
    # clues['rows'] = current_step.row_clues
    # clues['cols'] = current_step.col_clues

    for row_index, row_clues in enumerate(current_step.row_clues):
        options = all_options[str([clue for clue in row_clues if clue is not 0])]
        curr_row = current_step.matrix[row_index]

        for option in filter(lambda op: (op != curr_row).all(), options):
            if is_option_possible(option, curr_row):
                next_step = copy.deepcopy(current_step)
                next_step.matrix[row_index] = option
                next_steps.append(next_step)

    for col_index, col_clues in enumerate(current_step.col_clues):
        options = all_options[str([clue for clue in col_clues if clue is not 0])]
        curr_col = current_step.matrix[:, col_index]
        for option in filter(lambda op: (op != curr_col).all(), options):
            if is_option_possible(option, curr_col):
                next_step = copy.deepcopy(current_step)
                next_step.matrix[:, col_index] = option
                next_steps.append(next_step)

    return next_steps


def generate_next_steps_blocks(current_step):
    next_steps = []
    clues = dict()
    clues['rows'] = current_step.row_clues
    clues['cols'] = current_step.col_clues

    for row in range(len(current_step.matrix)):
        options = []
        for clue_index in range(len(clues['rows'][row])):
            clue = clues['rows'][row][clue_index]
            if is_block_exist(current_step.matrix[row], clue) == False:
                options = _generate_block(current_step.matrix[row], clue, clue_index * 2)

                for option in options:
                    next_step = copy.deepcopy(current_step)
                    next_step.matrix[row] = option
                    next_steps.append(next_step)

    for col in range(len(current_step.matrix[0])):
        for clue_index in range(len(clues['cols'][col])):
            clue = clues['cols'][col][clue_index]
            if is_block_exist(current_step.matrix[:, col], clue) == False:
                options = _generate_block(current_step.matrix[:, col], clue, clue_index * 2)

                for option in options:
                    next_step = copy.deepcopy(current_step)
                    next_step.matrix[:, col] = option
                    if validate_board(next_step.matrix, clues) == True:
                        next_steps.append(next_step)
                    else:
                        del next_step

    return next_steps


def _generate_block(row, size, min_index):
    options = []
    for cell in range(len(row) - size + 1 - min_index):
        if row[cell + min_index] == False:
            option = copy.copy(row)
            option[cell + min_index] = True
            for i in range(1, size):
                option[cell + i + min_index] = True

            options.append(option)
    return options


def is_block_exist(row, size):
    index = 0
    row_len = len(row)
    while index < row_len:
        clue_len = size
        while index < row_len and row[index] == False:
            index += 1

        while clue_len > 0 and index < row_len and row[index] == True:
            clue_len -= 1
            index += 1

        if index == row_len and clue_len == 0:
            return True

        if clue_len == 0 and (index <= row_len - 1 and row[index] == False):
            return True

    return False


def _check_block(row, clues_list):
    index = 0
    row_len = len(row)
    for clue in clues_list:

        if clue == 0:
            continue

        # Looking for the 1
        while index < row_len and row[index] == False:
            index += 1

        # Checking if the current block is longer than the clue
        clue_len = clue

        while clue_len > 0 and index < row_len and row[index] == True:
            clue_len -= 1
            index += 1

        if index == row_len:
            return True

        # If the block is longer than the clue
        if clue_len == 0 and row[index] == True:
            return False

    # We are out of clues
    while index < row_len and row[index] == False:
        index += 1

    if index == row_len:
        return True
    else:
        return False


def validate_board(candidate, clues):
    # checking the rows
    for row in range(len(candidate)):
        if _check_block(candidate[row], clues['rows'][row]) is False:
            return False

    # checking the cols
    for col in range(len(candidate[0])):
        if _check_block(candidate[:, col], clues['cols'][col]) is False:
            return False

    return True
