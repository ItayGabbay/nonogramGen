from nonogram import Nonogram
from utils import load_all_row_opts
from typing import List

CNF = List[List[bool]]


def nonogram_to_sat(nonogram: Nonogram) -> (CNF, CNF):
    all_opts = load_all_row_opts()
    row_clues = nonogram.row_clues
    # TODO use evaluator