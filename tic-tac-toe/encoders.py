from enum import IntEnum
from operator import itemgetter
import numpy as np
import numpy.typing as npt
from constants import BOARD_SIZE, WIDTH, HEIGHT
from math import floor
from engine import State
from printing import square_owner


# Assumes board shape is n x m i.e. no staggered arrays
def one_d_to_2_d(i, board_width):
    # type: (int, int) -> tuple[int, int]
    return (floor(i / board_width), i % board_width) # (row, column)

class OnePlaneEncoding(IntEnum):
    NOT_CURRENT_PLAYER = -1
    EMPTY = 0
    CURRENT_PLAYER = 1

def state_to_one_plane_encoding(state):
    # type: (State) -> npt.NDArray[np.int]
    # If the value of OnePlaneEncoding.EMPTY changes,
    # may need to modify this initialisation.
    # Keep for now as numpy.zeros() is a fast initialising function
    encoded_board = np.zeros((HEIGHT, WIDTH), dtype=int)
    bitboards, player_to_move = itemgetter('board', 'player_to_move')(state)

    for i in range(BOARD_SIZE):
        r, c = one_d_to_2_d(i, WIDTH)
        square = square_owner(i, bitboards)

        if square == player_to_move:
            encoded_board[r][c] = OnePlaneEncoding.CURRENT_PLAYER
        elif square != -1: # from square_owner(), -1 = empty square
            encoded_board[r][c] = OnePlaneEncoding.NOT_CURRENT_PLAYER

    return encoded_board

def one_hot_encode_move(move):
    # type: (int) -> npt.NDArray[np.int]
    encoded = np.zeros((HEIGHT, WIDTH), dtype=int)

    for i in range(BOARD_SIZE):
        r, c = one_d_to_2_d(i, WIDTH)
        square = (move >> i) & 1

        if square == 1:
            encoded[r][c] = 1

    return encoded
