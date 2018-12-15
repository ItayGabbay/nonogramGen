# NUM_ROWS = 20
NUM_ROWS = 5
# NUM_COLS = 20
NUM_COLS = 5
empty_in_split = '\xa0'

pickle_file_path = 'data/%dx%d_nonograms.pkl' % (NUM_ROWS, NUM_COLS)

NUM_COND_TREES = 5
NUM_VAL_TREES = NUM_COND_TREES + 1
prob_crossover_global = 0.5  # global probability for cx
prob_crossover_individual_cond = 1  # probability to cx a specific cond tree in an individual
prob_crossover_individual_val = 1  # probability to cx a specific cond tree in an individual
prob_mutate_global = 0.5
prob_mutate_individual_cond = 1
prob_mutate_individual_val = 1
