import operator
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from config import NUM_COND_TREES, NUM_VAL_TREES
import copy


def _make_condition_tree_pset():
    cond_pset = gp.PrimitiveSet("MAIN", 1)
    cond_pset.addPrimitive(operator.and_, 2)
    cond_pset.addPrimitive(operator.or_, 2)
    cond_pset.addPrimitive(operator.le, 2)
    cond_pset.addPrimitive(operator.ge, 2)
    cond_pset.renameArguments(ARG0='x')
    return cond_pset


def _make_value_tree_pset():
    val_pset = gp.PrimitiveSet("MAIN", 1)
    val_pset.addPrimitive(operator.add, 2)
    val_pset.addPrimitive(operator.mul, 2)
    val_pset.renameArguments(ARG0='x')
    return val_pset


def _init_individual(cls, cond_tree, val_tree):
    cond_trees = tools.initRepeat(list, cond_tree, NUM_COND_TREES)
    value_trees = tools.initRepeat(list, val_tree, NUM_VAL_TREES)

    return cls({"CONDITION_TREES": cond_trees, "VALUE_TREES": value_trees})


def make_toolbox(cond_pset: gp.PrimitiveSet, val_pset: gp.PrimitiveSet):
    toolbox = base.Toolbox()
    toolbox.register("value_expr", gp.genHalfAndHalf, pset=val_pset, min_=1, max_=2)
    toolbox.register("cond_expr", gp.genHalfAndHalf, pset=cond_pset, min_=1, max_=2)
    toolbox.register("value_tree", tools.initIterate, creator.ValueTree, toolbox.value_expr)
    toolbox.register("cond_tree", tools.initIterate, creator.ConditionTree, toolbox.cond_expr)
    toolbox.register("compile_valtree", gp.compile, pset=val_pset)
    toolbox.register("compile_condtree", gp.compile, pset=cond_pset)
    toolbox.register("evaluate", just_for_debug)  # just so an eval func will be defined
    toolbox.register("individual", _init_individual, creator.Individual, toolbox.cond_tree, toolbox.value_tree)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    return toolbox


def just_for_debug(individual):
    print(individual)
# def _make_cond_trees(num_of_trees:int = NUM_COND_TREES):
#     pset = _make_condition_tree_pset()
#     toolbox = _make_toolbox(pset)
#     pop = toolbox.population(n=num_of_trees)
#     return pop


def init_creator(cond_pset, val_pset):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("ValueTree", gp.PrimitiveTree, pset=val_pset)
    creator.create("ConditionTree", gp.PrimitiveTree, pset=cond_pset)
    creator.create("Individual", dict, fitness=creator.FitnessMin)


class TreeBasedIndividual(object):
    def __init__(self) -> None:
        cond_pset = _make_condition_tree_pset()
        val_pset = _make_value_tree_pset()

        init_creator(cond_pset, val_pset)
        toolbox = make_toolbox(cond_pset, val_pset)
        pop = toolbox.population(n=5)
        hof = tools.HallOfFame(1)
        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40,
                                       halloffame=hof, verbose=True)
        # TODO make val trees
