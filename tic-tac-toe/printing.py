from math import log10, floor
from constants import BOARD_SIZE, WIDTH, PLAYER_PIECE_SYMBOLS


def print_game_state(bitboards):
    # type: (list[int]) -> None
    for player, bitboard in enumerate(bitboards):
        print(f'Player {player}: {"{0:09b}".format(bitboard)}')

'''
  Returns the player that occupies the square at index
  If no player occupies the square at index, return -1
  If a collision is present i.e. if for some reason multiple player's bitboards
  occupy the same square, use the first player found
'''
# TO-DO: consider returning None instead of -1 for no square owner
def square_owner(index, bitboards):
    # type: (int, list[int]) -> int
    def loop(i, owner):
        # type: (int, int) -> int
        if owner != -1 or i >= len(bitboards):
            return owner
        else:
            return loop(i + 1,
                        i if (1 << index) & bitboards[i] > 0 else owner)

    return loop(0, -1)

'''
  Converts the game state (list of bitboards for each player)
  to a string of characters representing the board
  e.g. game_state_to_string(9, ['O', 'X'], [0b110001010, 0b001010101])
  returns 'XOXOX-XOO' (read left to right)
'''
def game_state_to_string(board_size, player_piece_symbols, bitboards):
    # type: (int, list[str], list[int]) -> str
    def loop(i, result):
        # type: (int, str) -> str
        if i >= board_size:
            return result

        square = square_owner(i, bitboards)

        return loop(i + 1,
                    result + (player_piece_symbols[square] if square >= 0 else '-'))

    return loop(0, '')

num_extra_padding = floor(log10(BOARD_SIZE))

def create_dash_string(n):
    # type (int) -> str
    def loop(dash_string, i):
        # type: (str, int) -> str
        if i == 0:
            return dash_string

        return loop(dash_string + '-', i - 1)

    return loop('-', n - 1)

def create_row_separator(n_col, dash_size):
    # type: (int, int) -> str
    dash_string = create_dash_string(dash_size)

    def loop(row_separator, i):
        # type: (str, int) -> str
        if i == 0:
            return row_separator + dash_string

        return loop(row_separator + dash_string + '+', i - 1)

    return loop('', n_col - 1)

row_separator = create_row_separator(BOARD_SIZE // WIDTH, num_extra_padding + 3)

'''
  Uses a chess-based indexing system,
  i.e. print the rows in reverse order
  e.g. for a standard tic-tac-toe board, coordinates look like:
   6 | 7 | 8
  ---+---+---
   3 | 4 | 5
  ---+---+---
   0 | 1 | 2
  Warning: I haven't tested this for other board dimensions
'''
def print_board(board_size, width, bitboards):
    # type: (int, int, list[int]) -> None
    n_rows = (board_size - 1) // width
    board_string = game_state_to_string(board_size, PLAYER_PIECE_SYMBOLS, bitboards)

    for row in reversed(range(n_rows + 1)):
        for i in range(row * width, row * width + width):
            current_piece = board_string[i]

            print((' ' if i == row * width else ' | ') + current_piece, end='')
        print() # newline

        if row > 0:
            print(row_separator)
