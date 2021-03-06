# # import operator
# # import math
# # import random
# #
# # import numpy
# #
# # from deap import algorithms
# # from deap import base
# # from deap import creator
# # from deap import tools
# # from deap import gp
# #
# #
# # # Define new functions
# # def protectedDiv(left, right):
# #     try:
# #         return left / right
# #     except ZeroDivisionError:
# #         return 1
# #
# #
# # pset = gp.PrimitiveSet("MAIN", 1)
# # pset.addPrimitive(operator.add, 2)
# # pset.addPrimitive(operator.sub, 2)
# # pset.addPrimitive(operator.mul, 2)
# # pset.addPrimitive(protectedDiv, 2)
# # pset.addPrimitive(operator.neg, 1)
# # pset.addPrimitive(math.cos, 1)
# # pset.addPrimitive(math.sin, 1)
# # pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
# # pset.renameArguments(ARG0='x')
# #
# # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# # creator.create("Individual", gp.PrimitiveTree)
# # creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
# #
# # toolbox = base.Toolbox()
# # toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
# # toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# # toolbox.register("compile", gp.compile, pset=pset)
# #
# #
# # # def evalSymbReg(individual, points):
# # #     # Transform the tree expression in a callable function
# # #     func = toolbox.compile(expr=individual)
# # #     # Evaluate the mean squared error between the expression
# # #     # and the real function : x**4 + x**3 + x**2 + x
# # #     sqerrors = ((func(x) - x ** 4 - x ** 3 - x ** 2 - x) ** 2 for x in points)
# # #     return math.fsum(sqerrors) / len(points),
# #
# #
# # toolbox.register("evaluate", lambda _: (1,))
# # # toolbox.register("evaluate", evalSymbReg, points=[x / 10. for x in range(-10, 10)])
# # # toolbox.register("select", tools.selTournament, tournsize=3)
# # # toolbox.register("mate", gp.cxOnePoint)
# # # toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
# # # toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
# #
# # # toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
# # # toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
# #
# #
# # def main():
# #     random.seed(318)
# #
# #     pop = toolbox.population(n=1)
# #     hof = tools.HallOfFame(1)
# #
# #     stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
# #     stats_size = tools.Statistics(len)
# #     mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
# #     mstats.register("avg", numpy.mean)
# #     mstats.register("std", numpy.std)
# #     mstats.register("min", numpy.min)
# #     mstats.register("max", numpy.max)
# #
# #     pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 0, stats=mstats,
# #                                    halloffame=hof, verbose=True)
# #     # print log
# #     return pop, log, hof
# #
# #
# # if __name__ == "__main__":
# #     main()
# import utils
# from nonogram import Nonogram
# # s = ' , , , ,10, , ,1,2,1, , ,1,2,1, , ,1,2,1, , , , ,16, , , ,1,1, , ,1,16,1, , , ,4,4, , , ,1,1, , , ,1,1, ,1,1,1,1, , , ,2,2,1,1,2,1,1, ,3,2,2,3, ,3,1,1,3, , ,4,4,4, , , ,6,6, ,2,5,4,3, ,4,2,1,5, , , ,9,9'
# # # s = ' , , , , , , , , , , , ,1, , , , , , , , , , , , ,1, , , , , , ,1,1,1, , , , , , , , , ,4,1,5,5,1,1,1,1,1,1,1,4, , , , , , ,1,1,3,1,1,1,1,1,1,1,1,1,1,3, ,1, , , ,1,2,2,1,4,1,1,1,1,1,1,2,1,4,1,1,2,1, ,1,1,1,5,5,4,3,2,1,1,1,1,2,3,4,4,2,1,1,1,2,3,4,2,1,1,1,4,3,1,1,3,1,1,1,2,8,4,3,2'
# # n = Nonogram(s, s)
# # print(n)
#
# ns = utils.load_nonograms_from_file()
# print(ns)
import utils
from GPExperiment import GPExperiment
import logging
import time

from plotter import Plotter
from config import plot_d3_fitness, plot_fitness_stats, plot_min_max_stats, plot_fitness_distribution_2d


def main():
    logging.basicConfig(filename='log/log_gp' + str(time.time()) + '.log',level=logging.DEBUG)
    gp = GPExperiment()
    logging.info('\n\n*******STARTING!!!******\n\n')
    logging.info('\n\n*******Configuration******\n\n')
    with open('./config.py', 'r') as f:
        config = f.readlines()
    for line in config:
        logging.info(line + '\n')
    pop, log, hof, stats, elapsed_time = gp.start_experiment()
    logging.info('\n\n*******DONE!!!******\n\n')
    logging.info('run time: %d sec\n', elapsed_time)
    logging.info('max possible fitness for the nonograms ran: %d\n')
    logging.info('log: %s\n', log)
    logging.info('pop: %s\n', utils.individual_lst_to_str(pop))
    logging.info('hof: %s\n', utils.individual_lst_to_str(hof))

    logging.info('stats: %s\n', stats)

    fitnesses = [ind.fitness.values for ind in pop]
    plot = Plotter(log, fitnesses)
    if plot_d3_fitness:
        plot.plot_population_tuples_3d()
    if plot_fitness_stats:
        plot.plot_fitness_stats_from_logbook()
    if plot_min_max_stats:
        plot.plot_min_max_counts()
    if plot_fitness_distribution_2d:
        plot.plot_fitness_distribution_2d()

from scoop import shared
from utils import load_train_and_test_sets
if __name__ == '__main__':
    train_test_sets = load_train_and_test_sets()
    train_dicts = train_test_sets['train']
    train_nonograms = [(d['unsolved'], d['solved']) for d in train_dicts]
    test_dicts = train_test_sets['test']
    test_nonograms = [(d['unsolved'], d['solved']) for d in test_dicts]
    shared.setConst(train_nonograms=train_nonograms)
    main()
