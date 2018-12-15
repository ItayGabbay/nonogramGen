import unittest
from utils import load_nonograms_from_file
import copy
from config import pickle_file_path
import heuristics

class TestHeuristics(unittest.TestCase):

    def setUp(self):
        self.nonograms = load_nonograms_from_file('../' + pickle_file_path)
        self.bomb_nonogram_unsolved = self.nonograms[0]
        self.bomb_nonogram_solved = copy.deepcopy(self.bomb_nonogram_unsolved)
        self.solve_bomb_nonogram()

    def solve_bomb_nonogram(self):
        mat = self.bomb_nonogram_solved.matrix
        mat[0][2] = True
        mat[0][3] = True

        mat[1][1] = True
        mat[1][4] = True

        mat[2][0] = True
        mat[2][1] = True
        mat[2][2] = True

        mat[3][0] = True
        mat[3][2] = True
        mat[3][2] = True

        mat[4][0] = True
        mat[4][1] = True
        mat[4][2] = True
        print(self.bomb_nonogram_solved)

    def test_ones_diff_rows_solved(self):
        actual = heuristics.ones_diff_rows(self.bomb_nonogram_solved)
        self.assertEqual(1.0, actual)

    def test_ones_diff_rows_unsolved(self):
        actual = heuristics.ones_diff_rows(self.bomb_nonogram_unsolved)
        self.assertEqual((25 - 12) / 25, actual)

    def test_ones_diff_rows_1_correct_filled(self):
        nonogram = copy.deepcopy(self.bomb_nonogram_unsolved)
        nonogram.matrix[0][2] = True
        actual = heuristics.ones_diff_rows(nonogram)
        self.assertEqual((25 - 11) / 25, actual)

    def test_ones_diff_rows_1_incorrect_filled(self):
        nonogram = copy.deepcopy(self.bomb_nonogram_unsolved)
        nonogram.matrix[0][1] = True
        actual = heuristics.ones_diff_rows(nonogram)
        self.assertEqual((25 - 11) / 25, actual)

    def test_ones_diff_cols_solved(self):
        actual = heuristics.ones_diff_cols(self.bomb_nonogram_solved)
        self.assertEqual(1.0, actual)

    def test_ones_diff_cols_unsolved(self):
        actual = heuristics.ones_diff_cols(self.bomb_nonogram_unsolved)
        self.assertEqual((25 - 12) / 25, actual)

    def test_ones_diff_cols_1_correct_filled(self):
        nonogram = copy.deepcopy(self.bomb_nonogram_unsolved)
        nonogram.matrix[0][2] = True
        actual = heuristics.ones_diff_cols(nonogram)
        self.assertEqual((25 - 11) / 25, actual)

    def test_ones_diff_cols_1_incorrect_filled(self):
        nonogram = copy.deepcopy(self.bomb_nonogram_unsolved)
        nonogram.matrix[0][1] = True
        actual = heuristics.ones_diff_cols(nonogram)
        self.assertEqual((25 - 11) / 25, actual)

    def test_zeros_diff_rows_solved(self):
        actual = heuristics.zeros_diff_rows(self.bomb_nonogram_solved)
        self.assertEqual(1.0, actual)

    def test_zeros_diff_rows_unsolved(self):
        actual = heuristics.zeros_diff_rows(self.bomb_nonogram_unsolved)
        self.assertEqual((25 - 12) / 25, actual)

    def test_zeros_diff_rows_1_correct_filled(self):
        nonogram = copy.deepcopy(self.bomb_nonogram_unsolved)
        nonogram.matrix[0][2] = True
        actual = heuristics.zeros_diff_rows(nonogram)
        self.assertEqual((25 - 11) / 25, actual)

    def test_zeros_diff_rows_1_incorrect_filled(self):
        nonogram = copy.deepcopy(self.bomb_nonogram_unsolved)
        nonogram.matrix[0][1] = True
        actual = heuristics.zeros_diff_rows(nonogram)
        self.assertEqual((25 - 11) / 25, actual)

if __name__ == '__main__':
    unittest.main()