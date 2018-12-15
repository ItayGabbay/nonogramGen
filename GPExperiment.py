import operator
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from config import NUM_COND_TREES, NUM_VAL_TREES
from heuristics import *
from evaluator import *
from utils import load_nonograms_from_file
from config import pickle_file_path
import random

nonograms = load_nonograms_from_file(path=pickle_file_path)

def _make_condition_tree_pset():
    def if_then_else(input, output1, output2):
        return output1 if input else output2

    cond_pset = gp.PrimitiveSetTyped("MAIN", [float, float, float, float, float, float], bool)
    cond_pset.addPrimitive(operator.__and__, [bool, bool], bool)
    cond_pset.addPrimitive(operator.__or__, [bool, bool], bool)
    cond_pset.addPrimitive(operator.le, [float, float], bool)
    cond_pset.addPrimitive(operator.ge, [float, float], bool)
    cond_pset.addEphemeralConstant('ephemeral_float', lambda: random.uniform(-5, -5), float)
    cond_pset.addEphemeralConstant('ephemeral_bool', lambda: random.choice([True, False]), bool)
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
    return toolbox


def evaluate(compile_valtree, compile_condtree, individual):
    compiled_conditions = [compile_condtree(cond_tree) for cond_tree in individual["CONDITION_TREES"]]
    compiled_values = [compile_valtree(val_tree) for val_tree in individual["VALUE_TREES"]]
    results = []
    for nonogram in nonograms:
        print(nonogram.title)
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
                        break;
                if heuristic is None:
                    heuristic = compiled_values[-1](ones_diff_rows_val,
                                                    ones_diff_cols_val,
                                                    zeros_diff_rows_val,
                                                    zeros_diff_cols_val,
                                                    compare_blocks_rows_val,
                                                    compare_blocks_cols_val)
                heuristics.append(heuristic)
            min_heuristic_index = heuristics.index(min(heuristics))
            selected_step = next_steps[min_heuristic_index]
            next_steps = generate_next_steps(selected_step)

        # Here need to compare to the solution!
        print(selected_step.matrix)
        results.append(10)
    return results

def init_creator(cond_pset, val_pset):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("ValueTree", gp.PrimitiveTree, pset=val_pset)
    creator.create("ConditionTree", gp.PrimitiveTree, pset=cond_pset)
    creator.create("Individual", dict, fitness=creator.FitnessMin)


class GPExperiment(object):
    def __init__(self) -> None:
        cond_pset = _make_condition_tree_pset()
        val_pset = _make_value_tree_pset()

        init_creator(cond_pset, val_pset)
        self.toolbox = make_toolbox(cond_pset, val_pset)
        self.pop = self.toolbox.population(n=5)
        self.hof = tools.HallOfFame(1)

    def start_experiment(self):
        pop, log = algorithms.eaSimple(self.pop, self.toolbox, 0.5, 0.1, 40,
                                       halloffame=self.hof, verbose=True)

