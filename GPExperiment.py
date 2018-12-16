import operator
from random import random, choice, uniform
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from config import *
from heuristics import *
from evaluator import *
from utils import load_nonograms_from_file
from config import pickle_unsolved_file_path, points_correct_box, points_incorrect_box
import numpy as np
from typing import Callable, Dict

nonograms = load_nonograms_from_file(path=pickle_unsolved_file_path)
def _make_condition_tree_pset():
    def if_then_else(input, output1, output2):
        return output1 if input else output2

    cond_pset = gp.PrimitiveSetTyped("MAIN", [float, float, float, float, float, float], bool)
    cond_pset.addPrimitive(operator.__and__, [bool, bool], bool)
    cond_pset.addPrimitive(operator.__or__, [bool, bool], bool)
    cond_pset.addPrimitive(operator.le, [float, float], bool)
    cond_pset.addPrimitive(operator.ge, [float, float], bool)
    cond_pset.addEphemeralConstant('ephemeral_float', lambda: uniform(-5, -5), float)
    cond_pset.addEphemeralConstant('ephemeral_bool', lambda: choice([True, False]), bool)
    cond_pset.addPrimitive(if_then_else, [bool, float, float], float)
    cond_pset.renameArguments(ARG0='ones_diff_rows')
    cond_pset.renameArguments(ARG1='ones_diff_cols')
    cond_pset.renameArguments(ARG2='zeros_diff_rows')
    cond_pset.renameArguments(ARG3='zeros_diff_cols')
    cond_pset.renameArguments(ARG4='compare_blocks_rows')
    cond_pset.renameArguments(ARG5='compare_blocks_cols')

    return cond_pset


def _make_value_tree_pset():
    val_pset = gp.PrimitiveSet("MAIN", 6)
    val_pset.addPrimitive(operator.add, 2)
    val_pset.addPrimitive(operator.mul, 2)
    val_pset.renameArguments(ARG0='ones_diff_rows')
    val_pset.renameArguments(ARG1='ones_diff_cols')
    val_pset.renameArguments(ARG2='zeros_diff_rows')
    val_pset.renameArguments(ARG3='zeros_diff_cols')
    val_pset.renameArguments(ARG4='compare_blocks_rows')
    val_pset.renameArguments(ARG5='compare_blocks_cols')

    return val_pset


def _init_individual(cls, cond_tree, val_tree):
    cond_trees = tools.initRepeat(list, cond_tree, NUM_COND_TREES)
    value_trees = tools.initRepeat(list, val_tree, NUM_VAL_TREES)

    return cls({"CONDITION_TREES": cond_trees, "VALUE_TREES": value_trees})


def make_toolbox(cond_pset: gp.PrimitiveSet, val_pset: gp.PrimitiveSet):
    toolbox = base.Toolbox()
    toolbox.register("value_expr", gp.genHalfAndHalf, pset=val_pset, min_=3, max_=5)
    toolbox.register("cond_expr", gp.genHalfAndHalf, pset=cond_pset, min_=3, max_=5)
    toolbox.register("value_tree", tools.initIterate, creator.ValueTree, toolbox.value_expr)
    toolbox.register("cond_tree", tools.initIterate, creator.ConditionTree, toolbox.cond_expr)
    toolbox.register("compile_valtree", gp.compile, pset=val_pset)
    toolbox.register("compile_condtree", gp.compile, pset=cond_pset)
    toolbox.register("evaluate", evaluate, toolbox.compile_valtree, toolbox.compile_condtree)
    toolbox.register("individual", _init_individual, creator.Individual, toolbox.cond_tree, toolbox.value_tree)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", _crossover)
    toolbox.register("mutate", _mutate, cond_expr=toolbox.cond_expr, val_expr=toolbox.value_expr,
                     cond_pset=cond_pset, val_pset=val_pset)
    return toolbox

def _flip_coin() -> bool:
    res = random() > 0.5
    print('flip coin is:', res)
    return res

def _crossover(individual1: Dict, individual2: Dict):
    cond_trees1 = individual1['CONDITION_TREES']
    cond_trees2 = individual2['CONDITION_TREES']
    val_trees1 = individual1['VALUE_TREES']
    val_trees2 = individual2['VALUE_TREES']

    if _flip_coin():  # cx cond trees
        for index in range(len(cond_trees1)):
            if random() <= prob_crossover_individual_cond:
                print('doing cx on cond trees idx %d' %(index,))
                t1 = cond_trees1[index]
                t2 = cond_trees2[index]
                print('t1 is', t1)
                print('t2 is', t2)
                t1, t2 = gp.cxOnePoint(t1, t2)
                print('t1 is', t1)
                print('t2 is', t2)
                cond_trees1[index] = t1
                cond_trees2[index] = t2
    else:
        for index in range(len(val_trees1)):
            if random() <= prob_crossover_individual_val:
                print('doing cx on val trees idx %d' %(index,))
                t1 = val_trees1[index]
                t2 = val_trees2[index]
                t1, t2 = gp.cxOnePoint(t1, t2)
                val_trees1[index] = t1
                val_trees2[index] = t2
    print('done cx')
    return individual1, individual2

def _mutate(individual: Dict, cond_expr, val_expr, cond_pset, val_pset):
    print('mutating', individual)
    if _flip_coin():
        expr = cond_expr
        pset = cond_pset
        prob = prob_mutate_individual_cond
        trees = individual['CONDITION_TREES']
    else:
        expr = val_expr
        pset = val_pset
        prob = prob_mutate_individual_val
        trees = individual['VALUE_TREES']
    for i, tree in enumerate(trees):
        if random() <= prob:
            tree, = gp.mutUniform(tree, expr, pset)
            trees[i] = tree
    return individual,

def _compare_to_solution(nonogram: Nonogram) -> int:
    def mapper_func(f: Callable[[bool, bool], bool], solved_mat: np.ndarray, check_mat: np.ndarray):
        inner_func = lambda row_solved, row_check: np.fromiter(map(f, row_solved, row_check), dtype=bool)
        m = map(inner_func, solved_mat, check_mat)
        return np.array(list(m))

    # TODO change this to be able to get any solved nonogram, not just Bomb
    solved = utils.load_bomb_nonogram_from_file().matrix
    to_check = nonogram.matrix
    corrects = mapper_func(lambda s, c: s and c, solved, to_check)
    wrongs = mapper_func(lambda s, c: (not s) and c, solved, to_check)
    res = points_correct_box * np.sum(corrects) - points_incorrect_box * np.sum(wrongs)
    return res if res >=0 else 0

def evaluate(compile_valtree, compile_condtree, individual):
    compiled_conditions = [compile_condtree(cond_tree) for cond_tree in individual["CONDITION_TREES"]]
    compiled_values = [compile_valtree(val_tree) for val_tree in individual["VALUE_TREES"]]
    results = []
    nonogram = utils.load_bomb_nonogram_from_file()
    results.append(evaluate_single_nonogram(compiled_conditions, compiled_values, nonogram))
    # TODO uncomment when we have solutions for all nonograms
    # for nonogram in utils.load_nonograms_from_file():
    #     results.append(evaluate_single_nonogram(compiled_conditions, compiled_values, nonogram))
    #     # results.append(np.sum(selected_step.matrix))
    return results


def evaluate_single_nonogram(compiled_conditions, compiled_values, nonogram):
    # print(nonogram.title)
    selected_step = nonogram
    next_steps = generate_next_steps(selected_step)
    while len(next_steps) > 0:
        heuristics = []
        # Evaluating the heuristics on the candidates and choosing the best
        for option in next_steps:
            ones_diff_rows_val = ones_diff_rows(option)
            ones_diff_cols_val = ones_diff_cols(option)
            zeros_diff_rows_val = zeros_diff_rows(option)
            zeros_diff_cols_val = zeros_diff_cols(option)
            compare_blocks_rows_val = compare_blocks_rows(option)
            compare_blocks_cols_val = compare_blocks_cols(option)
            heuristic = None
            for condition_index in range(len(compiled_conditions)):
                res = compiled_conditions[condition_index](ones_diff_rows_val,
                                                           ones_diff_cols_val,
                                                           zeros_diff_rows_val,
                                                           zeros_diff_cols_val,
                                                           compare_blocks_rows_val,
                                                           compare_blocks_cols_val)
                if res is True:
                    heuristic = compiled_values[condition_index](ones_diff_rows_val,
                                                                 ones_diff_cols_val,
                                                                 zeros_diff_rows_val,
                                                                 zeros_diff_cols_val,
                                                                 compare_blocks_rows_val,
                                                                 compare_blocks_cols_val)
                    break
            if heuristic is None:
                heuristic = compiled_values[-1](ones_diff_rows_val,
                                                ones_diff_cols_val,
                                                zeros_diff_rows_val,
                                                zeros_diff_cols_val,
                                                compare_blocks_rows_val,
                                                compare_blocks_cols_val)
            heuristics.append(heuristic)

        # heuristics are max based (the bigger the result the better)
        max_heuristic_index = heuristics.index(max(heuristics))
        selected_step = next_steps[max_heuristic_index]
        next_steps = generate_next_steps(selected_step)
    # Here need to compare to the solution!
    # print(selected_step.matrix)
    return _compare_to_solution(selected_step)


def init_creator(cond_pset, val_pset):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # TODO change this to be length of nonograms list
    creator.create("ValueTree", gp.PrimitiveTree, pset=val_pset)
    creator.create("ConditionTree", gp.PrimitiveTree, pset=cond_pset)
    creator.create("Individual", dict, fitness=creator.FitnessMax)


class GPExperiment(object):
    def __init__(self) -> None:
        cond_pset = _make_condition_tree_pset()
        val_pset = _make_value_tree_pset()

        init_creator(cond_pset, val_pset)
        self.toolbox = make_toolbox(cond_pset, val_pset)
        self.pop = self.toolbox.population(n=50)
        self.hof = tools.HallOfFame(1)

    def start_experiment(self):
        pop, log = algorithms.eaSimple(self.pop, self.toolbox, 0.5, 0.1, 40,
                                       halloffame=self.hof, verbose=True)
        return pop, log, self.hof

