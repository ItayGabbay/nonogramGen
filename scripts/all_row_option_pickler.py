from itertools import chain, combinations
from config import NUM_ROWS, pickle_row_options_path
import numpy as np
import pickle


def get_all_idx_options(line_len=NUM_ROWS):
    def powerset(xs):
        return list(chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1)))
    return powerset(range(line_len))


options = []
idxs = get_all_idx_options()
for idx_opt in idxs:
    row = np.full(NUM_ROWS, False)
    for idx in idx_opt:
        row[idx] = True
    options.append(row)

with open('../' + pickle_row_options_path, 'wb') as f:
    pickle.dump(options, f)
