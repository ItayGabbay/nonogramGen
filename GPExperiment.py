import operator
import time
from random import random
from typing import Callable, Dict

import numpy as np
from sys import stderr
from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools

from algorithms import eaSimple_new
from config import *
from evaluator import *
from heuristics import *
from individual import DoubleTreeBasedIndividual
from utils import load_train_and_test_sets
import pickle
from scoop import futures, shared


if should_run_in_parallel:
    train_nonograms = shared.getConst('train_nonograms')
    stderr.write("RUNNING WITH SCOOP! MAKE SURE YOU ARE RUNNING WITH: 'python -m scoop playground.py' !!\n")
    # shared.setConst(train=train_nonograms)

else:
    train_test_sets = load_train_and_test_sets()
    train_dicts = train_test_sets['train']
    train_nonograms = [(d['unsolved'], d['solved']) for d in train_dicts]
    test_dicts = train_test_sets['test']
    test_nonograms = [(d['unsolved'], d['solved']) for d in test_dicts]
def _if_then_else(inp, out1, out2):
    return out1 if inp else out2


# condition pset:
cond_pset = gp.PrimitiveSetTyped("MAIN", [float, float, float, float, float, float, float, float], bool)
cond_pset.addPrimitive(operator.__and__, [bool, bool], bool)
cond_pset.addPrimitive(operator.__or__, [bool, bool], bool)
cond_pset.addPrimitive(operator.le, [float, float], bool)
cond_pset.addPrimitive(operator.ge, [float, float], bool)
# cond_pset.addTerminal(True, bool)
# cond_pset.addTerminal(False, bool)
cond_pset.addEphemeralConstant("rand101", lambda: np.random.randint(0, 5), float)
cond_pset.addEphemeralConstant("randbool", lambda: np.random.choice([True, False]), bool)
cond_pset.addPrimitive(_if_then_else, [bool, float, float], float)
cond_pset.renameArguments(ARG0='ones_diff_rows')
cond_pset.renameArguments(ARG1='ones_diff_cols')
cond_pset.renameArguments(ARG2='zeros_diff_rows')
cond_pset.renameArguments(ARG3='zeros_diff_cols')
cond_pset.renameArguments(ARG4='compare_blocks_rows')
cond_pset.renameArguments(ARG5='compare_blocks_cols')
cond_pset.renameArguments(ARG6='max_row_clue')
cond_pset.renameArguments(ARG7='max_col_clue')

# value pset:
val_pset = gp.PrimitiveSet("MAIN", 8)
val_pset.addPrimitive(operator.add, 2)
val_pset.addPrimitive(operator.mul, 2)
val_pset.renameArguments(ARG0='ones_diff_rows')
val_pset.renameArguments(ARG1='ones_diff_cols')
val_pset.renameArguments(ARG2='zeros_diff_rows')
val_pset.renameArguments(ARG3='zeros_diff_cols')
val_pset.renameArguments(ARG4='compare_blocks_rows')
val_pset.renameArguments(ARG5='compare_blocks_cols')
val_pset.renameArguments(ARG6='max_row_clue')
val_pset.renameArguments(ARG7='max_col_clue')
val_pset.addEphemeralConstant("rand101_1", lambda : np.random.randint(0, 5))

# creator stuff:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("ValueTree", gp.PrimitiveTree, pset=val_pset)
creator.create("ConditionTree", gp.PrimitiveTree, pset=cond_pset)
creator.create("Individual", DoubleTreeBasedIndividual, fitness=creator.FitnessMin)


# nonograms = load_unsolved_nonograms_from_file(path=pickle_unsolved_file_path)


# def _make_condition_tree_pset():
#     cond_pset = gp.PrimitiveSetTyped("MAIN", [float, float, float, float, float, float], bool)
#     cond_pset.addPrimitive(operator.__and__, [bool, bool], bool)
#     cond_pset.addPrimitive(operator.__or__, [bool, bool], bool)
#     cond_pset.addPrimitive(operator.le, [float, float], bool)
#     cond_pset.addPrimitive(operator.ge, [float, float], bool)
#     cond_pset.addTerminal(True, bool)
#     cond_pset.addTerminal(False, bool)
#     for i in range(-5, 5, 1):
#         cond_pset.addTerminal(i, float)
#     cond_pset.addPrimitive(_if_then_else, [bool, float, float], float)
#     cond_pset.renameArguments(ARG0='ones_diff_rows')
#     cond_pset.renameArguments(ARG1='ones_diff_cols')
#     cond_pset.renameArguments(ARG2='zeros_diff_rows')
#     cond_pset.renameArguments(ARG3='zeros_diff_cols')
#     cond_pset.renameArguments(ARG4='compare_blocks_rows')
#     cond_pset.renameArguments(ARG5='compare_blocks_cols')
#
#     return cond_pset


# def _make_value_tree_pset():
#     val_pset = gp.PrimitiveSet("MAIN", 6)
#     val_pset.addPrimitive(operator.add, 2)
#     val_pset.addPrimitive(operator.mul, 2)
#     val_pset.renameArguments(ARG0='ones_diff_rows')
#     val_pset.renameArguments(ARG1='ones_diff_cols')
#     val_pset.renameArguments(ARG2='zeros_diff_rows')
#     val_pset.renameArguments(ARG3='zeros_diff_cols')
#     val_pset.renameArguments(ARG4='compare_blocks_rows')
#     val_pset.renameArguments(ARG5='compare_blocks_cols')
#
#     return val_pset


def _init_individual(cond_tree, val_tree, fitness=creator.FitnessMin()):
    return DoubleTreeBasedIndividual(cond_tree, val_tree, fitness)

# def _init_individual(cls, cond_tree, val_tree):
#     cond_trees = tools.initRepeat(list, cond_tree, NUM_COND_TREES)
#     value_trees = tools.initRepeat(list, val_tree, NUM_VAL_TREES)
#
#     return cls({"CONDITION_TREES": cond_trees, "VALUE_TREES": value_trees})


def make_toolbox(cond_pset_arg: gp.PrimitiveSetTyped = cond_pset, val_pset_arg: gp.PrimitiveSetTyped = val_pset):
    toolbox = base.Toolbox()
    toolbox.register("value_expr", gp.genHalfAndHalf, pset=val_pset_arg, min_=5, max_=5)
    toolbox.register("cond_expr", gp.genHalfAndHalf, pset=cond_pset_arg, min_=5, max_=5)
    toolbox.register("value_tree", tools.initIterate, creator.ValueTree, toolbox.value_expr)
    toolbox.register("cond_tree", tools.initIterate, creator.ConditionTree, toolbox.cond_expr)
    toolbox.register("compile_valtree", gp.compile, pset=val_pset_arg)
    toolbox.register("compile_condtree", gp.compile, pset=cond_pset_arg)
    toolbox.register("evaluate", evaluate, toolbox.compile_valtree, toolbox.compile_condtree)
    toolbox.register("individual", _init_individual, toolbox.cond_tree, toolbox.value_tree)
    # toolbox.register("individual", _init_individual, creator.Individual, toolbox.cond_tree, toolbox.value_tree)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.3, fitness_first=True)
    toolbox.register("mate", _crossover)
    toolbox.register("mutate", _mutate, cond_expr=toolbox.cond_expr, val_expr=toolbox.value_expr,
                     cond_pset=cond_pset_arg, val_pset=val_pset_arg)
    if should_run_in_parallel:
        print('running in parallel')
        toolbox.register("map", futures.map)

    return toolbox


def _flip_coin() -> bool:
    res = random() > 0.5
    # print('flip coin is:', res)
    return res


def _crossover(individual1: DoubleTreeBasedIndividual, individual2: DoubleTreeBasedIndividual):
    # cond_trees1 = individual1['CONDITION_TREES']
    cond_trees1 = individual1.cond_trees
    cond_trees2 = individual2.cond_trees
    # cond_trees2 = individual2['CONDITION_TREES']
    val_trees1 = individual1.cond_trees
    val_trees2 = individual2.cond_trees
    # val_trees1 = individual1['VALUE_TREES']
    # val_trees2 = individual2['VALUE_TREES']

    if _flip_coin():  # cx cond trees
        for index in range(len(cond_trees1)):
            if random() <= prob_crossover_individual_cond:
                # print('doing cx on cond trees idx %d' %(index,))
                t1 = cond_trees1[index]
                t2 = cond_trees2[index]
                # print('t1 is', t1)
                # print('t2 is', t2)
                t1, t2 = gp.cxOnePoint(t1, t2)
                # print('t1 is', t1)
                # print('t2 is', t2)
                cond_trees1[index] = t1
                cond_trees2[index] = t2
    else:
        for index in range(len(val_trees1)):
            if random() <= prob_crossover_individual_val:
                # print('doing cx on val trees idx %d' %(index,))
                t1 = val_trees1[index]
                t2 = val_trees2[index]
                t1, t2 = gp.cxOnePoint(t1, t2)
                val_trees1[index] = t1
                val_trees2[index] = t2
    # print('done cx')
    return individual1, individual2


def _mutate(individual: DoubleTreeBasedIndividual, cond_expr, val_expr, cond_pset, val_pset):
    # print('mutating', individual)
    if _flip_coin():
        expr = cond_expr
        pset = cond_pset
        prob = prob_mutate_individual_cond
        trees = individual.cond_trees
        # trees = individual['CONDITION_TREES']
    else:
        expr = val_expr
        pset = val_pset
        prob = prob_mutate_individual_val
        trees = individual.value_trees
        # trees = individual['VALUE_TREES']
    for i, tree in enumerate(trees):
        if random() <= prob:
            # tree, = gp.mutUniform(tree, expr, pset)
            tree, = gp.mutInsert(tree, pset)
            trees[i] = tree
    return individual,


def _calc_max_possible_fitness():
    compares = [compare_to_solution(solved, solved) for unsolved, solved in train_nonograms]
    res = np.mean(compares)
    print('max possible fitness is:', res)
    return res


def _calc_distance(fitnesses, to_compare_to=tuple(0 for _ in range(train_size))):
    pows = [np.power(fitnesses[i] - to_compare_to[i], 2) for i in range(len(fitnesses))]
    return np.sqrt(np.sum(pows))


def evaluate(compile_valtree, compile_condtree, individual: DoubleTreeBasedIndividual):
    compiled_conditions = [compile_condtree(cond_tree) for cond_tree in individual.cond_trees]
    # compiled_conditions = [compile_condtree(cond_tree) for cond_tree in individual["CONDITION_TREES"]]
    compiled_values = [compile_valtree(val_tree) for val_tree in individual.value_trees]
    # compiled_values = [compile_valtree(val_tree) for val_tree in individual["VALUE_TREES"]]
    results = []
    num_of_solved = 0
    # run with Bomb only
    # nonogram_solved = utils.load_solved_bomb_nonogram_from_file()
    # nonogram_unsolved = utils.load_unsolved_bomb_nonogram_from_file()
    # results.append(evaluate_single_nonogram(compiled_conditions, compiled_values, nonogram_solved, nonogram_unsolved))

    # run on all solved nonograms
    for nonogram_unsolved, nonogram_solved in train_nonograms:
        result = round(evaluate_single_nonogram(compiled_conditions, compiled_values, nonogram_solved, nonogram_unsolved), 4)
        results.append(result)
        if result < 1000:
            num_of_solved += 1
        # if result == 5:
        #     num_of_solved += 1
    # print('-------------------')

    # if num_of_solved > 0:
    #     print("Solved:", num_of_solved, "Nonograms")
    distance = _calc_distance(results)
    if print_individual_fitness:
        print("Fitness:", [round(res, 4) for res in results], round(distance, 4), 'solved:', num_of_solved)
    return distance,


def evaluate_single_nonogram(compiled_conditions, compiled_values, nonogram_solved: Nonogram,
                             nonogram_unsolved: Nonogram):
    result = perform_astar(compiled_conditions, compiled_values, nonogram_solved, nonogram_unsolved)
    # fitness_by_sat = selected_step.convert_to_sat()
    # fitness_by_compare = compare_to_solution(selected_step, nonogram_solved)
    # fitness = fitness_by_compare * fitness_by_sat
    # print('fitness:', fitness)
    # fitness = fitness_by_compare
    # print(nonogram_unsolved.title,  result)
    #if result < 1000:
        #print("Solved the", nonogram_unsolved.title, "with", result)
    return result


# TODO was pulled to global scope
# def init_creator(cond_pset, val_pset):
#     creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#     creator.create("ValueTree", gp.PrimitiveTree, pset=val_pset)
#     creator.create("ConditionTree", gp.PrimitiveTree, pset=cond_pset)
#     creator.create("Individual", dict, fitness=creator.FitnessMax)

def num_of_max(population):
    max_val = np.max(population)
    return len(list(filter(lambda i: i == max_val, population)))


def num_of_min(population):
    min_val = np.min(population)
    return len(list(filter(lambda i: i == min_val, population)))


def most_common(population):
    d = dict()
    for i in population:
        if i in d:
            d[i] = d[i] + 1
        else:
            d[i] = 1
    m = 0
    res = 0
    for fit, count in d.items():
        if count > m:
            res = fit
    if m == 1:
        return None
    return res[0]


class GPExperiment(object):
    def __init__(self) -> None:
        # cond_pset = _make_condition_tree_pset()
        # val_pset = _make_value_tree_pset()

        # init_creator(cond_pset, val_pset)
        self.toolbox = make_toolbox()
        self.pop = self.toolbox.population(n=pop_size)
        self.hof = tools.HallOfFame(hof_size)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        # stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit,)
        # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("median", np.median)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        mstats.register("size", len)
        mstats.register("num max", num_of_max)
        mstats.register("num min", num_of_min)
        mstats.register("most common", most_common)
        self.stats = mstats

    def start_experiment(self):
        nonogram_names = [unsolved.title for unsolved, solved in train_nonograms]
        print('running experiment on', train_size, 'nonograms. names:', nonogram_names)
        # max_possible_fitness = _calc_max_possible_fitness()

        start = time.time()
        # mu = len(self.pop)
        # lambda_ = len(self.pop)
        # pop, log = algorithms.eaMuPlusLambda(self.pop, self.toolbox, mu, lambda_, prob_crossover_global, prob_mutate_global, num_gen,
        #                                      halloffame=self.hof, verbose=True, stats=self.stats)
        pop, log = eaSimple_new(self.pop, self.toolbox, prob_crossover_global, prob_mutate_global, num_gen,
                                       halloffame=self.hof, verbose=True, stats=self.stats)
        end = time.time()
        return pop, log, self.hof, self.stats, end - start
