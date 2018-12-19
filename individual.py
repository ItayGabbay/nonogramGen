from deap import tools
from config import NUM_COND_TREES, NUM_VAL_TREES
from typing import List


class DoubleTreeBasedIndividual(dict):
    def __init__(self, cond_tree, val_tree, **kwargs):
        super().__init__(**kwargs)
        self.cond_trees: List = tools.initRepeat(list, cond_tree, NUM_COND_TREES)
        self.value_trees: List = tools.initRepeat(list, val_tree, NUM_VAL_TREES)

    def __len__(self):
        return len(self.cond_trees) + len(self.value_trees)
