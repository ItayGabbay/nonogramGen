from itertools import chain, combinations
from config import NUM_ROWS, pickle_row_options_path, all_rows_archive_name
import numpy as np
import pickle
from typing import List, Dict
from klepto.archives import dir_archive


def get_all_idx_options(line_len=NUM_ROWS):
    def powerset(xs):
        return list(chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1)))

    return powerset(range(line_len))


def find_clue_for_row(row_to_check: List[bool]) -> List[int]:
    def skip_till_val(start=0, val=True):
        if start >= len(row_to_check):
            return len(row_to_check), -1

        count = 0
        while start < len(row_to_check) and row_to_check[start] != val:
            start += 1
            count += 1
        return start, count

    clues_to_ret = []
    block_idx, block_len = skip_till_val()
    while block_idx < len(row_to_check):
        block_idx, block_len = skip_till_val(block_idx, False)
        clues_to_ret.append(block_len)
        block_idx, block_len = skip_till_val(block_idx)
    return clues_to_ret


def add_or_append_to_dict(clue_to_rows: Dict[str, List], row_to_add, clue_as_str: str):
    if clue_as_str in clue_to_rows:
        clue_to_rows[clue_as_str].append(row_to_add)
    else:
        clue_to_rows[clue_as_str] = [row_to_add]
    return clue_to_rows


options = []
idxs = get_all_idx_options()
for idx_opt in idxs:
    row = np.full(NUM_ROWS, False)
    for idx in idx_opt:
        row[idx] = True
    options.append(row)

print('found %d options' % (len(options)))

d = dict()
for opt in options:
    clues = find_clue_for_row(opt)
    print('found clues for', opt, clues)
    d = add_or_append_to_dict(d, opt, str(clues))

archive = dir_archive('../' + all_rows_archive_name, d, serialized=True)
# save
archive.dump()
del archive

# with open('../' + pickle_row_options_path, 'wb') as f:
#     pickle.dump(d, f)
