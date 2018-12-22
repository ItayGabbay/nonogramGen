# NUM_ROWS = 20
NUM_ROWS = 20
# NUM_COLS = 20
NUM_COLS = 20
empty_in_split = '\xa0'

pickle_unsolved_file_path = './data/%dx%d_nonograms.pkl' % (NUM_ROWS, NUM_COLS)
pickle_solved_file_path = './data/%dx%d_nonograms_solved.pkl' % (NUM_ROWS, NUM_COLS)
pickle_row_options_path = './data/%dx%d_row_options.pkl' % (NUM_ROWS, NUM_ROWS)
should_run_in_parallel = True
print_individual_fitness = True
convert_to_sat = True

NUM_COND_TREES = 5
NUM_VAL_TREES = NUM_COND_TREES + 1
prob_crossover_global = 1  # global probability for cx
# prob_crossover_global = 0.7  # global probability for cx
prob_crossover_individual_cond = 0.8  # probability to cx a specific cond tree in an individual
prob_crossover_individual_val = 0.8  # probability to cx a specific cond tree in an individual
prob_mutate_global = 0.3
# prob_mutate_global = 1
# prob_mutate_individual_cond = 1
prob_mutate_individual_cond = 0.7
# prob_mutate_individual_val = 1
prob_mutate_individual_val = 0.7

points_correct_box = 5
points_incorrect_box = 0
# points_incorrect_box = -2
pop_size = 150
hof_size = 1
# num_gen = 40
num_gen = 10
train_size = 3
