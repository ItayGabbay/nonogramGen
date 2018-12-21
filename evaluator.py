import copy


def evaluate_individual(individual, step):
    next_steps = generate_next_steps(step)

    # while len(next_steps) > 0:
    #     print "AAA"


def generate_next_steps(current_step):
    next_steps = []
    clues = dict()
    clues['rows'] = current_step.row_clues
    clues['cols'] = current_step.col_clues

    for row in range(len(current_step.matrix)):
        for col in range(len(current_step.matrix[0])):
            candidate = current_step.matrix.copy()

            if candidate[row][col] == False:
                candidate[row][col] = True
                if validate_board(candidate, clues) is True:
                    new_nonogram = copy.deepcopy(current_step)
                    new_nonogram.matrix = candidate
                    next_steps.append(new_nonogram)

    return next_steps


def _check_block(row, clues_list):
    index = 0
    row_len = len(row)
    for clue in clues_list:

        if clue == 0:
            continue

        # Looking for the 1
        while index < row_len and row[index] == False:
            index += 1

        # Checking if the current block is longer than the clue
        clue_len = clue

        while clue_len > 0 and index < row_len and row[index] == True:
            clue_len -= 1
            index += 1

        if index == row_len:
            return True

        # If the block is longer than the clue
        if clue_len == 0 and row[index] == True:
            return False

    # We are out of clues
    while index < row_len and row[index] == False:
        index += 1

    if index == row_len:
        return True
    else:
        return False


def validate_board(candidate, clues):
    # checking the rows
    for row in range(len(candidate)):
        if _check_block(candidate[row], clues['rows'][row]) is False:
            return False

    # checking the cols
    for col in range(len(candidate[0])):
        if _check_block(candidate[:, col], clues['cols'][col]) is False:
            return False

    return True
