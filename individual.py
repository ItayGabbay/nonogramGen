from deap import tools, base, gp
from config import NUM_COND_TREES, NUM_VAL_TREES
from typing import List
import numpy as np


class DoubleTreeBasedIndividual(dict):
    def __init__(self, cond_tree: gp.PrimitiveTree, val_tree: gp.PrimitiveTree, fitness: base.Fitness, **kwargs):
        super().__init__(**kwargs)
        self.cond_trees: List[gp.PrimitiveTree] = tools.initRepeat(list, cond_tree, NUM_COND_TREES)
        self.value_trees: List[gp.PrimitiveTree] = tools.initRepeat(list, val_tree, NUM_VAL_TREES)
        self.fitness = fitness

    def __len__(self):
        sum_func = lambda trees: np.mean([tree.height for tree in trees])
        return int(sum_func(self.cond_trees) + sum_func(self.value_trees))