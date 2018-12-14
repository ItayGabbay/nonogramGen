from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

class TreeBasedIndividual(object):
    def __init__(self) -> None:
        # TODO create the tree lists here
        pass

    def _make_condition_tree_pset(self):
        cond_pset = gp.PrimitiveSet("MAIN", 1)

