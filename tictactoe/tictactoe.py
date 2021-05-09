"""
Tic Tac Toe Player
"""

import math
import copy , random , numpy as np
X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    board1 = np.array(board)
    x_value = np.where(board1 == X)
    o_value = np.where(board1 == O)
    if len(x_value[0]) > len(o_value[0]):
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    board1 = np.array(board)
    action = np.where(board1 == None)
    x = action[0]
    y = action[1]
    action = [(x[i], y[i]) for i in range(len(x))]
    return action


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    borad1 = copy.deepcopy(board)
    if action not in actions(board):
        raise NameError("Not A Valid Action")
    else:
        borad1[action[0]][action[1]] = player(board)
        return borad1


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    board1 = np.array(board)
    if np.any(np.all(board1 == X, axis=1)) or np.any(np.all(board1 == X, axis=0)):
        return X
    elif np.any(np.all(board1 == O, axis=1)) or np.any(np.all(board1 == O, axis=0)):
        return O
    elif (board1[0, 0] == X and board1[1, 1] == X and board1[2, 2] == X) or (
            board1[0, 2] == X and board1[1, 1] == X and board1[2, 0] == X):
        return X
    elif (board1[0, 0] == O and board1[1, 1] == O and board1[2, 2] == O) or (
            board1[0, 2] == O and board1[1, 1] == O and board1[2, 0] == O):
        return O
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) == None:
        board1 = np.array(board)
        none_value = np.where(board1 == None)
        if len(none_value[0]) > 0:
            return False
        else:
            return True
    else:
        return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    corners = list(set([(0, 0), (0, 2), (2, 0), (2, 2)]) & set(actions(board)))
    counter_move = None
    if terminal(board):
        return None
    elif player(board) == X:
        if (1, 1) in actions(board):
            return 1, 1
        for act in actions(board):
            board1 = result(board, act)
            if winner(board1) == X:
                return act
            for next_act in actions(board1):
                if utility(result(board1, next_act)) == -1:
                    counter_move = next_act
                    break
        else:
            if counter_move is not None:
                return counter_move
            elif not corners:
                return random.choice(actions(board))
            else:
                return random.choice(corners)
    else:
        if (1, 1) in actions(board):
            return 1, 1
        for act in actions(board):
            board1 = result(board, act)
            if winner(board1) == O:
                return act
            for next_act in actions(board1):
                if utility(result(board1, next_act)) == 1:
                    counter_move = next_act
                    break
        else:
            if counter_move is not None:
                return counter_move
            elif not corners:
                return random.choice(actions(board))
            else:
                return random.choice(corners)
