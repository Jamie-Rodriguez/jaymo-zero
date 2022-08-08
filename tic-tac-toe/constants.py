from math import floor


BOARD_SIZE = 9
WIDTH = 3 # number of columns
# Should always be an int anyway, but just in case we use floor()
HEIGHT = floor(BOARD_SIZE / WIDTH) # number of rows

PLAYER_PIECE_SYMBOLS = ['O', 'X']

BOARD_AREA = (1 << BOARD_SIZE) - 1

# TO-DO: find a way to generate this dynamically for arbitrary board dimensions
THREE_IN_A_ROW = [0b001001001,
                  0b010010010,
                  0b100100100,
                  0b000000111,
                  0b000111000,
                  0b111000000,
                  0b100010001,
                  0b001010100]

NEW_GAME = [0, 0]
