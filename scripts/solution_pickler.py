from utils import load_nonograms_from_file
from config import pickle_unsolved_file_path, pickle_solved_file_path
import pickle

# solve and save Bomb, TODO: change this to solve and load all nonograms

bomb = load_nonograms_from_file('../' + pickle_unsolved_file_path)[0]
mat = bomb.matrix
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
print(bomb)

with open('../' + pickle_solved_file_path, 'wb') as f:
    pickle.dump(bomb, f)

print('done')
