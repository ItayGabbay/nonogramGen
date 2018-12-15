import unittest
from config import pickle_file_path
from utils import load_nonograms_from_file
import evaluator


class EvaluatorTests(unittest.TestCase):

    def setUp(self):
        from_files = load_nonograms_from_file('../' + pickle_file_path)
        self.semi_nonogram = from_files[0]
        self.semi_nonogram.matrix[0] = [False, True, True, False, False]
        self.empty_nonogram = from_files[1]

    def test_semi_nonogram(self):
        actual = evaluator.generate_next_steps(self.semi_nonogram)
        self.assertEqual(19, len(actual))

    def test_empty_nonogram(self):
        actual = evaluator.generate_next_steps(self.empty_nonogram)
        self.assertEqual(25, len(actual))


if __name__ == '__main__':
    unittest.main()