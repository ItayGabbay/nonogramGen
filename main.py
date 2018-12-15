from utils import load_nonograms_from_file
from config import pickle_file_path
from GPExperiment import GPExperiment

nonograms = load_nonograms_from_file(path=pickle_file_path)
experiment = GPExperiment()
experiment.start_experiment()
