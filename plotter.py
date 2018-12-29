# This is a Harry Plotter
from os import mkdir, path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pygraphviz as pgv
from deap import gp
from deap.tools import Logbook, HallOfFame

from config import plots_dir_path, plot_img_format


def _check_if_dir_exists(dir_path=plots_dir_path):
    if not path.isdir(dir_path):
        mkdir(dir_path)


# for manual string plotting
def plot_tree_from_str(string, pset, i):
    def _make_graph(edges, labels, nodes):
        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")
        for node in nodes:
            n = g.get_node(node)
            n.attr["label"] = labels[node]
        return g

    tree = gp.PrimitiveTree.from_string(string, pset)
    nodes, edges, labels = gp.graph(tree)
    labels = {key: str(val).replace('_', '\n') for key, val in labels.items()}
    g = _make_graph(edges, labels, nodes)
    print('drawing', i)
    g.draw("./plots/fromLog/condTrees/" + str(i) + ".png")


def plot_lst_trees(pset, str_to_split: str):
    sp = str_to_split.split(';')
    for i, s in enumerate(sp):
        plot_tree_from_str(s, pset, i)


class Plotter(object):
    def __init__(self, logbook: Logbook, fitnesses: List[tuple], hof: HallOfFame):
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
        self.hof = hof

    def plot_population_tuples_3d(self, show_plot=True):
        _check_if_dir_exists()

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
        _check_if_dir_exists()

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
        _check_if_dir_exists()

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
        _check_if_dir_exists()

        gen = list(range(self.num_gen))
        plt.plot(gen, self.res_dict['num max'])
        plt.plot(gen, self.res_dict['num min'])
        plt.legend(['NUM MAX', 'NUM MIN'], loc='upper left')

        plt.savefig(plots_dir_path + '/min_max_counts.' + plot_img_format, bbox_inches='tight')
        if show_plot:
            plt.show(block=True)

    # def plot_hof_trees(self):
    #     _check_if_dir_exists()
    #
    #     hof: HallOfFame = self.hof
    #     _check_if_dir_exists(plots_dir_path + '/hof')
    #
    #     cond_dir = plots_dir_path + '/hof/cond'
    #     val_dir = plots_dir_path + '/hof/val'
    #     _check_if_dir_exists(cond_dir)
    #     _check_if_dir_exists(val_dir)
    #
    #     for ind_idx, ind in enumerate(hof.items):
    #         for i, cont_tree in enumerate(ind.cond_trees):
    #             nodes, edges, labels = gp.graph(cont_tree)
    #             g = nx.Graph()
    #             g.add_nodes_from(nodes)
    #             g.add_edges_from(edges)
    #             # pos = nx.graphviz_layout(g, prog="dot")
    #             pos = pygraphviz_layout(g)
    #
    #             nx.draw_networkx_nodes(g, pos)
    #             nx.draw_networkx_edges(g, pos)
    #             nx.draw_networkx_labels(g, pos, labels)
    #             plt.savefig(cond_dir + '/' + str(i) + '.' + plot_img_format, bbox_inches='tight')
    #
    #         for i, val_tree in enumerate(ind.cond_trees):
    #             nodes, edges, labels = gp.graph(val_tree)
    #             g = nx.Graph()
    #             g.add_nodes_from(nodes)
    #             g.add_edges_from(edges)
    #             # pos = nx.graphviz_layout(g, prog="dot")
    #             pos = pygraphviz_layout(g, prog='sfdp', root='0', args='-Lg')
    #
    #             nx.draw_networkx_nodes(g, pos)
    #             nx.draw_networkx_edges(g, pos)
    #             nx.draw_networkx_labels(g, pos, labels)
    #             plt.savefig(val_dir + '/' + str(i) + '.' + plot_img_format, bbox_inches='tight')

    def plot_hof_trees(self):
        def _make_graph(edges, labels, nodes):
            g = pgv.AGraph()
            g.add_nodes_from(nodes)
            g.add_edges_from(edges)
            g.layout(prog="dot")
            for node in nodes:
                n = g.get_node(node)
                n.attr["label"] = labels[node]
            return g

        _check_if_dir_exists()
        hof: HallOfFame = self.hof
        _check_if_dir_exists(plots_dir_path + '/hof')

        cond_dir = plots_dir_path + '/hof/cond'
        val_dir = plots_dir_path + '/hof/val'
        _check_if_dir_exists(cond_dir)
        _check_if_dir_exists(val_dir)
        for ind_idx, ind in enumerate(hof.items):
            for i, cont_tree in enumerate(ind.cond_trees):
                nodes, edges, labels = gp.graph(cont_tree)
                labels = {key: str(val).replace('_', '\n') for key, val in labels.items()}
                g = _make_graph(edges, labels, nodes)

                g.draw(cond_dir + "/" + str(i) + '.' + plot_img_format)

            for i, val_tree in enumerate(ind.value_trees):
                nodes, edges, labels = gp.graph(val_tree)
                labels = {key: str(val).replace('_', '\n') for key, val in labels.items()}
                g = _make_graph(edges, labels, nodes)

                g.draw(val_dir + "/" + str(i) + "." + plot_img_format)
