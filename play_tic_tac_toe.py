from tictactoe.engine import (
    get_valid_moves_list,
    is_terminal,
    apply_move_to_state,
    check_win,
    play_game,
    play_n_games,
    make_random_agent
)
from tictactoe.constants import NEW_GAME
from mcts.mcts import make_mcts_agent


if __name__ == '__main__':
    print('-------------------------------- Demo one game ---------------------------------')
    play_game([make_mcts_agent(1.2,
                               get_valid_moves_list,
                               is_terminal,
                               apply_move_to_state,
                               check_win,
                               0, # Player index!
                               1000),
               make_random_agent(get_valid_moves_list)],
              [NEW_GAME])

    n = 100
    print(f'------------------------------ Playing {n} games ------------------------------')
    play_n_games([make_random_agent(get_valid_moves_list),
                  make_mcts_agent(1.2,
                                  get_valid_moves_list,
                                  is_terminal,
                                  apply_move_to_state,
                                  check_win,
                                  1, # Player index!
                                  100)],
                 n)
