from utils import load_unsolved_nonograms_from_file
from config import pickle_unsolved_file_path, pickle_solved_file_path
from typing import List
from nonogram import Nonogram
import pickle


def solve_bomb(bomb: Nonogram):
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
    return bomb

def solve_camel(camel: Nonogram):
    mat = camel.matrix
    mat[0][1] = True

    mat[1][0] = True
    mat[1][1] = True
    mat[1][2] = True
    mat[1][4] = True

    mat[2][0] = True
    mat[2][1] = True
    mat[2][2] = True
    mat[2][3] = True

    mat[3][0] = True
    mat[3][2] = True

    mat[4][0] = True
    mat[4][2] = True
    return camel

def solve_tetris(tetris: Nonogram):
    mat = tetris.matrix
    mat[0][2] = True

    mat[1][1] = True
    mat[1][2] = True

    mat[2][0] = True
    mat[2][2] = True
    mat[2][4] = True

    mat[3][0] = True
    mat[3][3] = True
    mat[3][4] = True

    mat[4][0] = True
    mat[4][1] = True
    mat[4][3] = True
    mat[4][4] = True
    return tetris

def solve_baby_carriage(baby_carriage: Nonogram):
    mat = baby_carriage.matrix
    mat[0][1] = True
    mat[0][2] = True

    mat[1][0] = True
    mat[1][1] = True
    mat[1][4] = True

    mat[2][0] = True
    mat[2][1] = True
    mat[2][2] = True
    mat[2][3] = True

    mat[3][1] = True
    mat[3][2] = True

    mat[4][0] = True
    mat[4][3] = True
    return baby_carriage

def solve_fountain(fountain: Nonogram):
    mat = fountain.matrix
    mat[0][1] = True
    mat[0][3] = True

    mat[1][0] = True
    mat[1][2] = True
    mat[1][4] = True

    mat[2][2] = True

    mat[3][0] = True
    mat[3][1] = True
    mat[3][2] = True
    mat[3][3] = True
    mat[3][4] = True

    mat[4][1] = True
    mat[4][2] = True
    mat[4][3] = True
    return fountain

def solve_rabbit(rabbit: Nonogram):
    mat = rabbit.matrix
    mat[0][1] = True
    mat[0][2] = True

    mat[1][3] = True
    mat[1][4] = True

    mat[2][1] = True
    mat[2][2] = True
    mat[2][3] = True
    mat[2][4] = True

    mat[3][0] = True
    mat[3][1] = True
    mat[3][2] = True
    mat[3][3] = True

    mat[4][1] = True
    mat[4][2] = True
    mat[4][3] = True
    mat[4][4] = True
    return rabbit

def solve_candle(candle: Nonogram):
    mat = candle.matrix
    mat[0][1] = True

    mat[2][1] = True

    mat[3][1] = True
    mat[3][3] = True
    mat[3][4] = True

    mat[4][0] = True
    mat[4][1] = True
    mat[4][2] = True
    mat[4][3] = True
    return candle

def solve_chicken(chicken: Nonogram):
    mat = chicken.matrix
    mat[0][1] = True

    mat[1][0] = True
    mat[1][1] = True
    mat[1][3] = True
    mat[1][4] = True

    mat[2][1] = True
    mat[2][2] = True
    mat[2][3] = True

    mat[3][1] = True
    mat[3][2] = True
    mat[3][3] = True

    mat[4][2] = True
    return chicken

def solve_dog(dog: Nonogram):
    mat = dog.matrix
    mat[0][3] = True
    mat[0][4] = True

    mat[1][0] = True
    mat[1][3] = True
    mat[1][4] = True

    mat[2][0] = True
    mat[2][1] = True
    mat[2][2] = True
    mat[2][3] = True

    mat[3][1] = True
    mat[3][3] = True

    mat[4][1] = True
    mat[4][3] = True
    return dog

def solve_hourglass(hourglass: Nonogram):
    mat = hourglass.matrix
    mat[0][0] = True
    mat[0][1] = True
    mat[0][2] = True
    mat[0][3] = True
    mat[0][4] = True

    mat[1][1] = True
    mat[1][2] = True
    mat[1][3] = True

    mat[2][2] = True

    mat[3][1] = True
    mat[3][3] = True

    mat[4][0] = True
    mat[4][1] = True
    mat[4][2] = True
    mat[4][3] = True
    mat[4][4] = True
    return hourglass

def solve_watch(watch: Nonogram):
    mat = watch.matrix
    mat[0][1] = True
    mat[0][2] = True
    mat[0][3] = True

    mat[1][0] = True
    mat[1][2] = True
    mat[1][3] = True

    mat[2][0] = True
    mat[2][2] = True
    mat[2][3] = True
    mat[2][4] = True

    mat[3][0] = True
    mat[3][4] = True

    mat[4][1] = True
    mat[4][2] = True
    mat[4][3] = True
    return watch

def solve_cat(cat: Nonogram):
    mat = cat.matrix
    mat[0][2] = True
    mat[0][4] = True

    mat[1][2] = True
    mat[1][3] = True
    mat[1][4] = True

    mat[2][0] = True
    mat[2][1] = True
    mat[2][2] = True
    mat[2][3] = True
    mat[2][4] = True

    mat[3][0] = True
    mat[3][1] = True
    mat[3][2] = True
    mat[3][3] = True

    mat[4][0] = True
    mat[4][1] = True
    mat[4][2] = True
    mat[4][3] = True
    mat[4][4] = True
    return cat

def solve_camel2(camel: Nonogram):
    mat = camel.matrix
    mat[0][3] = True
    mat[0][4] = True

    mat[1][0] = True
    mat[1][1] = True
    mat[1][3] = True

    mat[2][0] = True
    mat[2][1] = True
    mat[2][2] = True
    mat[2][3] = True

    mat[3][0] = True
    mat[3][2] = True

    mat[4][0] = True
    mat[4][2] = True
    return camel

def solve_tiny_heart(heart: Nonogram):
    mat = heart.matrix
    mat[0][1] = True
    mat[0][3] = True

    mat[1][0] = True
    mat[1][1] = True
    mat[1][2] = True
    mat[1][3] = True
    mat[1][4] = True

    mat[2][0] = True
    mat[2][1] = True
    mat[2][2] = True
    mat[2][3] = True
    mat[2][4] = True

    mat[3][1] = True
    mat[3][2] = True
    mat[3][3] = True

    mat[4][2] = True
    return heart

def save(nonos: List[Nonogram]):
    with open('../' + pickle_solved_file_path, 'wb') as f:
        pickle.dump(nonos, f)

def solve_nono(index: int, unsolved_nono: Nonogram):
    print('solving', unsolved_nono.title, unsolved_nono.number)
    solved = {0: solve_bomb,
              1: solve_camel,
              2: solve_tetris,
              3: solve_baby_carriage,
              4: solve_fountain,
              5: solve_rabbit,
              6: solve_candle,
              7: solve_chicken,
              8: solve_dog,
              9: solve_hourglass,
              10: solve_watch,
              11: solve_cat,
              12: solve_camel2,
              13: solve_tiny_heart,
              }[index](unsolved_nono)
    return solved

nonograms = load_unsolved_nonograms_from_file('../' + pickle_unsolved_file_path)
res = [solve_nono(i, n) for (i, n) in enumerate(nonograms)]
save(res)
print('done')
