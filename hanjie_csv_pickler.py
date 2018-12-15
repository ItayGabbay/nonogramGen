import csv
from config import NUM_COLS, NUM_ROWS, pickle_file_path
from nonogram import Nonogram
import pickle

csv_path = 'data/hanjie.csv'

save = True  # True -> save to file, False -> load file


# saves the Nonograms in csv_path to a pickled list of unsolved Nonograms
with open(csv_path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    filtered = [{'title': row[2], 'number': int(row[3]), 'solution': row[4], 'rows':row[6], 'cols': row[7]}
                for i, row in enumerate(readCSV) if i > 0 and int(row[0]) == NUM_COLS and int(row[1]) == NUM_ROWS]
    nonograms = [Nonogram(d['rows'], d['cols'], d['title'], d['number'], d['solution']) for d in filtered]

with open(pickle_file_path, 'wb') as f:
    pickle.dump(nonograms, f)