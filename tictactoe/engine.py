from typing import TypedDict, List, Callable
from functools import reduce
from operator import itemgetter
from random import randint
from tail_recursive import tail_recursive, FeatureSet
from .constants import BOARD_AREA, THREE_IN_A_ROW, BOARD_SIZE, WIDTH, NEW_GAME
from .printing import print_game_state, print_board


class State(TypedDict):
    board: List[int]
    player_to_move: int

'''
  Uses a modified version of Brian Kernighan's Algorithm.
  Instead of just counting the number of set bits, writes the set bits into a list.
  Uses the fact that n & (n-1) removes the *rightmost* SET bit from n
  Then use n XOR (n & (n-1)) to get the removed bit, and add it to the list
  e.g.:
      input: 1101
      output: [0001, 0100, 1000]
'''
def separate_bitboard(bitboard):
    # type: (int) -> list[int]
    def loop(n, list_of_bitboards):
        # type: (int, list[int]) -> list[int]
        if n == 0:
            return list_of_bitboards

        remove_rightmost_setbit =  n & (n - 1)

        return loop(remove_rightmost_setbit,
                    list_of_bitboards + [(n ^ remove_rightmost_setbit)])

    return loop(bitboard, [])

# Bitwise-and with the boundary of the board so that we don't have to deal with
# negative number arithmetic
def get_valid_moves_bitmask(bitboards):
    # type: (list[int]) -> int
    return BOARD_AREA & ~reduce(lambda a, b: a | b, bitboards)

# apply_move() is idempotent; a move on an already-occupied square or out of
# bounds should return the same state
def apply_move(bitboards, player, move):
    # type: (list[int], int, int) -> list[int]
    if (move & get_valid_moves_bitmask(bitboards)) > 0:
        return list(map(lambda bb, i:
                            bb if i != player else BOARD_AREA & (move | bitboards[player]),
                        bitboards,
                        range(len(bitboards))))
    return bitboards

def apply_move_to_state(state, move):
    # type: (State, int) -> State
    board, player_to_move = itemgetter('board', 'player_to_move')(state)

    return {
        'board': apply_move(board, player_to_move, move),
        'player_to_move': (player_to_move + 1) % len(board)
    }

def is_full(bitboards):
    # type: (list[int]) -> bool
    return (BOARD_AREA ^ reduce(lambda a, b: a | b, bitboards)) == 0

'''
   output |        meaning
  --------+------------------------
    None  | neither player has won
     0    | player 1 wins
     1    | player 2 wins
'''
def check_win(state):
    # type: (State) -> int | None
    def loop(player, win):
        # type: (int, int | None) -> int | None
        if player >= len(state['board']):
            return win

        if reduce(
            lambda result, direction:
                result or (direction == (state['board'][player] & direction)),
            THREE_IN_A_ROW,
            False):

            return loop(player + 1, player)

        return loop(player + 1, win)

    return loop(0, None)

def is_terminal(state):
    # type: (State) -> bool
    return is_full(state['board']) or (check_win(state) is not None)

def get_valid_moves_list(state):
    # type: (State) -> list[int]
    valid_moves_bitmask = get_valid_moves_bitmask(state['board'])

    return (separate_bitboard(valid_moves_bitmask)
           if (valid_moves_bitmask > 0) and not is_terminal(state)
           else [])


def make_random_agent(get_moves_list):
    # type: (Callable[[State], list[int] | None]) -> Callable[[State], int]
    def pick_random_move(state):
        # type: (State) -> int
        moves = get_moves_list(state)
        random_index = randint(0, len(moves) - 1)

        return moves[random_index]

    return pick_random_move


# Lots of visual information to help debugging
def play_game(agents, initial_board):
    # type: (list[Callable[[State], int]], list[int]) -> None
    def loop(history):
        # type: (list[list[int]]) -> None
        turn_number = len(history) - 1
        player_to_move = turn_number % len(agents)
        current_bitboards = history[turn_number]
        current_state = { 'board': current_bitboards,
                          'player_to_move': player_to_move }
        move = agents[player_to_move](current_state)
        new_bitboards = apply_move(current_bitboards, player_to_move, move)
                    # We don't actually use the new player_to_move
                    # Only including it for completeness' sake
        new_state = { 'board': new_bitboards,
                      'player_to_move': (turn_number + 1) % len(agents) }
        new_history = history + [new_bitboards]
        win_status = check_win(new_state)

        print(f'Turn {turn_number + 1}')
        print(f'Current player: {player_to_move}')
        print('New board:')
        print_game_state(new_bitboards)
        print() # newline
        if win_status is not None:
            print(f'Player {win_status} wins!')
        print() # newline
        print_board(BOARD_SIZE, WIDTH, new_bitboards)
        print() # newline
        if is_terminal(new_state):
            return new_history

        return loop(new_history)

    return loop(initial_board)

# No visual information, used for simulating a large amount of games
def play_game_result(agents, initial_board):
    # type: (list[Callable[[State], int]], list[int]) -> int | None
    def loop(history):
        # type: (list[list[int]]) -> int | None
        turn_number = len(history) - 1
        player_to_move = turn_number % len(agents)
        current_bitboards = history[turn_number]
        current_state = { 'board': current_bitboards,
                          'player_to_move': player_to_move }
        move = agents[player_to_move](current_state)
        new_bitboards = apply_move(current_bitboards, player_to_move, move)
                    # We don't actually use the new player_to_move
                    # Only including it for completeness' sake
        new_state = { 'board': new_bitboards,
                      'player_to_move': (turn_number + 1) % len(agents) }
        new_history = history + [new_bitboards]
        win_status = check_win(new_state)

        if is_terminal(new_state):
            return win_status

        return loop(new_history)

    return loop(initial_board)

def play_n_games(agents, num_games):
    # type: (list[Callable[[State], int]], int) -> dict
    stats = { 'wins': [0, 0], 'draws': 0 }

    for _ in range(num_games):
        result = play_game_result(agents, [NEW_GAME])

        stats = ({ **stats, 'draws': stats['draws'] + 1 } if result is None else
                 { **stats,
                   'wins': [w + 1 if i == result else w
                                for i, w in enumerate(stats['wins'])] })

    return print(stats)


if __name__ == '__main__':
    print('-------------------------------- Demo one game ---------------------------------')
    play_game([make_random_agent(get_valid_moves_list),
               make_random_agent(get_valid_moves_list)],
              [NEW_GAME])

    n = 900
    print(f'------------------------------ Playing {n} games ------------------------------')
    play_n_games([make_random_agent(get_valid_moves_list),
                  make_random_agent(get_valid_moves_list)],
                 n)
