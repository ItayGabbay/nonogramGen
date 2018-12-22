# This is a Harry Plotter
import matplotlib.pyplot as plt
from deap.tools import Logbook
from config import fitness_plot_path, nums_plot_path
import pickle


class Plotter(object):
    def __init__(self, logbook: Logbook):
        fitness_chapter = logbook.chapters['fitness']
        res_dict = dict()
        for d in fitness_chapter:
            for key, val in d.items():
                if key not in res_dict:
                    res_dict[key] = [val]
                else:
                    res_dict[key].append(val)
        self.res_dict = res_dict
        self.num_gen = len(fitness_chapter)

    def plot_fitness_values(self):
        gen = list(range(self.num_gen))
        plt.plot(gen, self.res_dict['avg'])
        plt.plot(gen, self.res_dict['median'])
        plt.plot(gen, self.res_dict['most common'])
        plt.plot(gen, self.res_dict['max'])
        plt.plot(gen, self.res_dict['min'])
        plt.legend(['avg', 'median', 'most common', 'MAX', 'MIN'], loc='upper left')

        graph = plt.show(block=True)

        with open(fitness_plot_path, 'wb') as f:
            pickle.dump(graph, f)

    def plot_sizes(self):
        gen = list(range(self.num_gen))
        plt.plot(gen, self.res_dict['num max'])
        plt.plot(gen, self.res_dict['num min'])
        plt.legend(['NUM MAX', 'NUM MIN'], loc='upper left')

        graph = plt.show(block=True)

        with open(fitness_plot_path, 'wb') as f:
            pickle.dump(graph, f)


