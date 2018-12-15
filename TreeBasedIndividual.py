import operator
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from config import NUM_COND_TREES
import copy


def _make_condition_tree_pset():
    cond_pset = gp.PrimitiveSet("MAIN", 1)
    cond_pset.addPrimitive(operator.and_, 2)
    cond_pset.addPrimitive(operator.or_, 2)
    cond_pset.addPrimitive(operator.le, 2)
    cond_pset.addPrimitive(operator.ge, 2)
    cond_pset.renameArguments(ARG0='x')
    return cond_pset


def _make_toolbox(pset: gp.PrimitiveSet):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", lambda _: (1,))  # just so an eval func will be defined
    return toolbox

def _make_cond_trees(num_of_trees:int = NUM_COND_TREES):
    pset = _make_condition_tree_pset()
    toolbox = _make_toolbox(pset)
    pop = toolbox.population(n=num_of_trees)
    return pop

def init_creator():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree)
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


class TreeBasedIndividual(object):
    def __init__(self) -> None:
        init_creator()
        self.cond_trees = _make_cond_trees()
        print(self.cond_trees)
        # TODO make val trees
