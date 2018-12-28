# This is a Harry Plotter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap.tools import Logbook
from config import plots_dir_path, plot_img_format
import pickle
from typing import List
import numpy as np
from os import mkdir, path


def _check_if_plot_dir_exists():
    if not path.isdir(plots_dir_path):
        mkdir(plots_dir_path)


class Plotter(object):
    def __init__(self, logbook: Logbook, fitnesses: List[tuple]):
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
        self.population_fitness = fitnesses

    def plot_population_tuples_3d(self, show_plot=True):
        _check_if_plot_dir_exists()

        count_dict = dict()
        for fit in self.population_fitness:
            if fit not in count_dict:
                count_dict[fit] = 1
            else:
                count_dict[fit] = count_dict[fit] + 1
        # count_dict = {fit: len(val) for fit, val in count_dict}
        fit1 = []
        fit2 = []
        fit3 = []
        counts = []

        for fit, count in count_dict.items():
            fit1.append(fit[0])
            fit2.append(fit[1])
            fit3.append(fit[2])
            counts.append(count)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.hot()
        ax.scatter(np.array(fit1), np.array(fit2), np.array(fit3), c=np.array(counts))
        fig.savefig(plots_dir_path + '/population_3d.' + plot_img_format, bbox_inches='tight')
        if show_plot:
            plt.show(block=True)

    def plot_fitness_distribution_2d(self, show_plot=True):
        _check_if_plot_dir_exists()

        count_dict = dict()
        for fit in self.population_fitness:
            if fit not in count_dict:
                count_dict[fit] = 1
            else:
                count_dict[fit] = count_dict[fit] + 1
        fit_lst = []
        counts = []

        for fit, count in count_dict.items():
            fit_lst.append(fit)
            counts.append(count)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(np.array(fit_lst), np.array(counts))
        plt.xlabel('fitness')
        plt.ylabel('count')

        fig.savefig(plots_dir_path + '/fitness_distrib_2d.' + plot_img_format, bbox_inches='tight')
        if show_plot:
            plt.show(block=True)

    def plot_fitness_stats_from_logbook(self, show_plot=True):
        _check_if_plot_dir_exists()

        gen = list(range(self.num_gen))
        plt.plot(gen, self.res_dict['avg'])
        plt.plot(gen, self.res_dict['median'])
        plt.plot(gen, self.res_dict['most common'])
        plt.plot(gen, self.res_dict['max'])
        plt.plot(gen, self.res_dict['min'])
        plt.legend(['avg', 'median', 'most common', 'MAX', 'MIN'], loc='upper left')

        plt.savefig(plots_dir_path + '/fitness_stats.' + plot_img_format, bbox_inches='tight')
        if show_plot:
            plt.show(block=True)

    def plot_min_max_counts(self, show_plot=True):
        _check_if_plot_dir_exists()

        gen = list(range(self.num_gen))
        plt.plot(gen, self.res_dict['num max'])
        plt.plot(gen, self.res_dict['num min'])
        plt.legend(['NUM MAX', 'NUM MIN'], loc='upper left')

        plt.savefig(plots_dir_path + '/min_max_counts.' + plot_img_format, bbox_inches='tight')
        if show_plot:
            plt.show(block=True)
