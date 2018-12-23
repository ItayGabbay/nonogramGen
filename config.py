import time

# NUM_ROWS = 20
NUM_ROWS = 5
# NUM_COLS = 20
NUM_COLS = 5
empty_in_split = '\xa0'
pickle_unsolved_file_path = 'data/%dx%d_nonograms.pkl' % (NUM_ROWS, NUM_COLS)
pickle_solved_file_path = 'data/%dx%d_nonograms_solved.pkl' % (NUM_ROWS, NUM_COLS)
pickle_row_options_path = 'data/%dx%d_row_options.pkl' % (NUM_ROWS, NUM_ROWS)
fitness_plot_path = 'plots/fitness' + str(time.time()) + '.pkl'
nums_plot_path = 'plots/nums' + str(time.time()) + '.pkl'
plot_fitness_distr_path = 'plots/fitneess_distr' + str(time.time()) + '.pkl'
plot_population_3d = 'plots/population_3d' + str(time.time()) + '.pkl'

convert_to_sat = True
should_run_in_parallel = True
print_individual_fitness = False

plot_fitness_stats = True
plot_min_max_stats = True
plot_d3_fitness = True
plot_fitness_distribution_2d = True

NUM_COND_TREES = 5
NUM_VAL_TREES = NUM_COND_TREES + 1
prob_crossover_global = 0.8  # global probability for cx
prob_crossover_individual_cond = 0.7  # probability to cx a specific cond tree in an individual
prob_crossover_individual_val = 0.7  # probability to cx a specific cond tree in an individual
prob_mutate_global = 0.2
prob_mutate_individual_cond = 1
prob_mutate_individual_val = 1

points_correct_box = 5
points_incorrect_box = 0
# points_incorrect_box = -2
pop_size = 100
hof_size = 1
# num_gen = 40
num_gen = 40
train_size = 3
