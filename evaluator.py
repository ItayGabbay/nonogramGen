import copy
from utils import load_all_row_opts
from typing import List
from nonogram import Nonogram
from heuristics import *
import numpy as np
import operator
from config import steps_threshold, points_correct_box, points_incorrect_box
from typing import Callable

def evaluate_individual(individual, step):
    next_steps = generate_next_steps(step)

    # while len(next_steps) > 0:
    #     print "AAA"

def perform_astar(compiled_conditions, compiled_values, nonogram_solved: Nonogram,
                  nonogram_unsolved: Nonogram):

    selected_step = dict({'nonogram': nonogram_unsolved, 'f_value': None, 'parent': None})
    closed_list = []
    number_of_steps = 0
    open_list = [selected_step]
    while np.allclose(nonogram_solved.matrix, selected_step['nonogram'].matrix) == False:

        if selected_step not in closed_list:
            new_nodes = generate_next_steps_blocks_options(selected_step['nonogram'])
            closed_list.append(selected_step)
            open_list.remove(selected_step)

            if len(new_nodes) == 0:
                selected_step = selected_step['parent']
            else:
                for node_index in range(len(new_nodes)):
                    open_list.append({'nonogram': new_nodes[node_index], 'f_value': None, 'parent': selected_step})

        # Evaluating the heuristics on the candidates and choosing the best
        for option in open_list:
            if option['f_value'] is None:
                ones_diff_rows_val = ones_diff_rows(option['nonogram'])
                ones_diff_cols_val = ones_diff_cols(option['nonogram'])
                zeros_diff_rows_val = zeros_diff_rows(option['nonogram'])
                zeros_diff_cols_val = zeros_diff_cols(option['nonogram'])
                compare_blocks_rows_val = compare_blocks_rows(option['nonogram'])
                compare_blocks_cols_val = compare_blocks_cols(option['nonogram'])
                max_row_clue = get_max_col_clue(option['nonogram'])
                max_col_clue = get_max_col_clue(option['nonogram'])
                heuristic = None

                # print('ones_diff_rows_val', ones_diff_rows_val)
                # print('ones_diff_cols_val', ones_diff_cols_val)
                # print('zeros_diff_rows_val', zeros_diff_rows_val)
                # print('zeros_diff_cols_val', zeros_diff_cols_val)
                # print('compare_blocks_rows_val', compare_blocks_rows_val)
                # print('compare_blocks_cols_val', compare_blocks_cols_val)
                # print('max_row_clue', max_row_clue)
                # print('max_col_clue', max_col_clue)

                for condition_index in range(len(compiled_conditions)):
                    res = compiled_conditions[condition_index](ones_diff_rows_val,
                                                               ones_diff_cols_val,
                                                               zeros_diff_rows_val,
                                                               zeros_diff_cols_val,
                                                               compare_blocks_rows_val,
                                                               compare_blocks_cols_val,
                                                               max_row_clue,
                                                               max_col_clue)
                    if res is True:
                        heuristic = compiled_values[condition_index](ones_diff_rows_val,
                                                                     ones_diff_cols_val,
                                                                     zeros_diff_rows_val,
                                                                     zeros_diff_cols_val,
                                                                     compare_blocks_rows_val,
                                                                     compare_blocks_cols_val,
                                                                     max_row_clue,
                                                                     max_col_clue
                                                                     )
                        break
                if heuristic is None:
                    heuristic = compiled_values[-1](ones_diff_rows_val,
                                                    ones_diff_cols_val,
                                                    zeros_diff_rows_val,
                                                    zeros_diff_cols_val,
                                                    compare_blocks_rows_val,
                                                    compare_blocks_cols_val,
                                                    max_row_clue,
                                                    max_col_clue
                                                    )
                option['f_value'] = heuristic

        # heuristics are max based (the bigger the result the better)
        children = [x for x in open_list if x['parent'] is selected_step]
        while len(children) == 0:
            selected_step = selected_step['parent']
            children = [x for x in open_list if x['parent'] is selected_step]
        max_heuristic = max(children, key=lambda x: x['f_value'])
        # max_heuristic_index = heuristics.index(max(heuristics))
        # print("Max heuristic:", max(heuristics), " index:", max_heuristic_index)
        selected_step = max_heuristic
        number_of_steps += 1
        if number_of_steps > steps_threshold:
            print("Reached", steps_threshold, "steps for", nonogram_solved.title)
            grade = compare_to_solution(selected_step['nonogram'], nonogram_solved)
            return steps_threshold * grade
        # print(selected_step.matrix)

        # next_steps = generate_next_steps_blocks(selected_step)
    return number_of_steps


def compare_to_solution(nonogram: Nonogram, nonogram_solved: Nonogram) -> int:
    def mapper_func(f: Callable[[bool, bool], bool], solved_mat: np.ndarray, check_mat: np.ndarray):
        inner_func = lambda row_solved, row_check: np.fromiter(map(f, row_solved, row_check), dtype=bool)
        m = map(inner_func, solved_mat, check_mat)
        return np.array(list(m))

    mat_solved = nonogram_solved.matrix
    to_check = nonogram.matrix
    corrects = mapper_func(lambda s, c: s and c, mat_solved, to_check)
    maximum = mapper_func(lambda s, c: s and c, mat_solved, mat_solved)
    wrongs = mapper_func(lambda s, c: (not s) and c, mat_solved, to_check)
    # print('nono', nonogram.title, 'corrects:', corrects, 'wrongs:', wrongs)
    max_sum = np.sum(maximum)
    whites_in_solution = (NUM_COLS * NUM_ROWS) - max_sum
    corrects_ratio = np.sum(corrects) / max_sum
    epsilon = 0.0001
    incorrects_ratio = np.sum(wrongs) / whites_in_solution
    # print('corrects:', corrects_ratio, 'incorrects:', incorrects_ratio)
    res = points_correct_box * corrects_ratio + points_incorrect_box * incorrects_ratio + epsilon

    # if max_sum == np.sum(corrects) and np.sum(wrongs) == 0:
    #     print("Solved the ", nonogram.title, "Fitness:", points_correct_box * np.sum(corrects))
    return max_sum - res

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

        for option in filter(lambda op: np.allclose(op, curr_row) == False, options):
            if is_option_possible(option, curr_row):
                next_step = copy.deepcopy(current_step)
                next_step.matrix[row_index] = option

                exists=False
                # Check if we don't have it already
                for mat_index in range(len(next_steps)):
                    if np.allclose(next_steps[mat_index].matrix, next_step.matrix):
                        exists=True
                        break;
                if exists == False:
                    next_steps.append(next_step)

    for col_index, col_clues in enumerate(current_step.col_clues):
        options = all_options[str([clue for clue in col_clues if clue is not 0])]
        curr_col = current_step.matrix[:, col_index]
        for option in filter(lambda op: (op != curr_col).all(), options):
            if is_option_possible(option, curr_col):
                next_step = copy.deepcopy(current_step)
                next_step.matrix[:, col_index] = option

                exists=False
                # Check if we don't have it already
                for mat_index in range(len(next_steps)):
                    if np.allclose(next_steps[mat_index].matrix, next_step.matrix):
                        exists=True
                        break;
                if exists == False:
                    next_steps.append(next_step)

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
                    if validate_board(next_step.matrix, clues) == True:
                        next_steps.append(next_step)
                    else:
                        del next_step

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
