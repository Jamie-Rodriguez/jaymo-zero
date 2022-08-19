from math import inf, exp
from functools import reduce
from random import seed, randint
from sys import maxsize
from mcts import (
    uct,
    pick_best_move,
    State,
    Move,
    pick_unexplored_move,
    select,
    treewalk,
    replace_node,
    simulate,
    is_path_valid,
    backprop
)


def test_uct():
    assert 2.5 == uct(1, # Exploration parameter
                      exp(8), # total rollouts on parent node
                      { 'num_rollouts': 2, 'score': 1 }) # Node statistics
    assert inf == uct(1, exp(8), { 'num_rollouts': 0, 'score': 0 })

def test_pick_best_move():
    assert 0 == pick_best_move(1.5, { 'moves': [{ 'num_rollouts': 2,
                                                  'score': 1 }]})
    assert 0 == pick_best_move(1.5, { 'moves': [{ 'num_rollouts': 2,
                                                  'score': 1 },
                                                { 'num_rollouts': 8,
                                                  'score': 3 } ]})
    assert 1 == pick_best_move(1.5, { 'moves': [{ 'num_rollouts': 2,
                                                  'score': 1 },
                                                { 'num_rollouts': 0,
                                                  'score': 0 }]})

def test_pick_unexplored_move():
    # "Random" number generator to pick a move from the valid moves
    def mock_random_int(): # type: () -> int
        return 1
    # Valid/unexplored moves generator
    def mock_get_valid_moves(state): # type: (State) -> list[Move]
        board = state['board']

        if board == [0b100010010, 0b011000100]:
            return [0b000100000, 0b000001000]

        if board == [0b001110001, 0b110001100]:
            return [0b000000010]

        return []

    def mock_is_terminal(state): # type: (State) -> bool
        return state['board'] == [0b001001101, 0b100010010]

    # Not fully explored
    assert 0b000001000 == pick_unexplored_move(
        mock_random_int,
        mock_get_valid_moves,
        mock_is_terminal,
        {
          # 'move': 0b100000000,
          'state': { 'board': [0b100010010, 0b011000100],
                     'player_to_move': 1 },
          # 'num_rollouts': 2,
          # 'score': 1,
          'moves': [{ 'move': 0b000000001,
                      # 'state': { 'board': [0b100010010, 0b011000101],
                      #            'player_to_move': 0 },
                      # 'num_rollouts': 2,
                      # 'score': 1,
                      # 'moves': []
                    }]})

    # Fully-explored
    # Selection should never pick a fully-explored node...
    assert None is pick_unexplored_move(
        mock_random_int,
        mock_get_valid_moves,
        mock_is_terminal,
        {
          # 'move': 0b010000000,
          'state': { 'board': [0b001110001, 0b110001100],
                     'player_to_move': 0 },
          # 'num_rollouts': 2,
          # 'score': 0,
          'moves': [{ 'move': 0b000000010,
                      # 'state': { 'board': [0b001110011, 0b110001100],
                      #            'player_to_move': 1 },
                      # 'num_rollouts': 1,
                      # 'score': 0,
                      # 'moves': []
                    }]})

    # Terminal state
    assert None is pick_unexplored_move(
        mock_random_int,
        mock_get_valid_moves,
        mock_is_terminal,
        {
          # 'move': 0b000001000,
          'state': { 'board': [0b001001101, 0b100010010],
                     'player_to_move': 1 },
          # 'num_rollouts': 2,
          # 'score': 1,
          'moves': []})

def test_select():
    '''
      Start from state { 'board': [0b010000101, 0b000011010], 'player_to_move': 0 }
      There are three possible moves to choose from.
    '''
    initial_state = { 'state': { 'board': [0b010000101, 0b000011010],
                                'player_to_move': 0 },
                      # 'move': 0b000001000,
                      'num_rollouts': 5,
                      # 'score': 3,
                      'moves': [] }

    exploration = 1.5

    def mock_tie_breaker(): # type: () -> int
        return 1

    '''
      Need to have two different outputs as the last test checks that
      selection drills further down the tree.
      Therefore also need to make sure that we can output the valid moves
      of the next state
    '''
    def mock_get_valid_moves(state): # type: (State) -> list[Move]
        if state['board'] == [0b010000101, 0b000011010]:
            return [0b000100000, 0b001000000, 0b100000000]

        return [0b000100000, 0b100000000]

    def mock_is_terminal(state): # type: (State) -> bool
        return False


    # Test case: no moves have been expanded for root
    # Select root node for expansion
    assert [] == select(exploration,
                        mock_tie_breaker,
                        mock_get_valid_moves,
                        mock_is_terminal,
                        initial_state)

    # Test case: only one move out of three has been expanded for root
    # Select root node for expansion
    assert [] == select(exploration,
                        mock_tie_breaker,
                        mock_get_valid_moves,
                        mock_is_terminal,
                        { **initial_state,
                          'moves': [{
                                # 'move': 0b000100000,
                                # 'state': {'board': [0b010100101, 0b000011010],
                                #           'player_to_move': 1},
                                # 'num_rollouts': 5,
                                # 'score': 3,
                                # 'moves': []
                        }]})

    # Test case: all three moves have been expanded for root,
    # with same statistics
    # Resolve tie-break for selection
    assert [0b001000000] == select(exploration,
                                   mock_tie_breaker,
                                   mock_get_valid_moves,
                                   mock_is_terminal,
                                   { **initial_state,
                                     'moves': [
                                           {
                                               # 'move': 0b000100000,
                                               # 'state': { 'board': [0b010100101, 0b000011010],
                                               #            'player_to_move': 1 },
                                               'num_rollouts': 1,
                                               'score': 1,
                                               # 'moves': []
                                           },
                                           {
                                               'move': 0b001000000,
                                               'state': { 'board': [0b011000101, 0b000011010],
                                                          'player_to_move': 1 },
                                               'num_rollouts': 1,
                                               'score': 1,
                                               'moves': []
                                           },
                                           {
                                               # 'move': 0b100000000,
                                               # 'state': { 'board': [0b110000101, 0b000011010],
                                               #            'player_to_move': 1 },
                                               'num_rollouts': 1,
                                               'score': 1,
                                               # 'moves': []
                                           }
                                   ]})

    # Test case: all three moves have been expanded for root,
    # with differing statistics
    # Select move with highest UCT value
    assert [0b001000000] == select(exploration,
                                   mock_tie_breaker,
                                   mock_get_valid_moves,
                                   mock_is_terminal,
                                   { **initial_state,
                                     'moves': [
                                            {
                                                # 'move': 0b000100000,
                                                # 'state': {'board': [0b010100101, 0b000011010],
                                                #           'player_to_move': 1},
                                                'num_rollouts': 1,
                                                'score': 0,
                                                # 'moves': []
                                            },
                                            {
                                                'move': 0b001000000,
                                                'state': {'board': [0b011000101, 0b000011010],
                                                          'player_to_move': 1},
                                                'num_rollouts': 2,
                                                'score': 2,
                                                'moves': []
                                            },
                                            {
                                                # 'move': 0b100000000,
                                                # 'state': {'board': [0b110000101, 0b000011010],
                                                #           'player_to_move': 1},
                                                'num_rollouts': 2,
                                                'score': 1,
                                                # 'moves': []
                                            }
                                    ]})

    # Test case: all three moves have been expanded for root,
    # with differing statistics
    # Select move with highest UCT value - which is fully expanded.
    # (i.e. not a leaf node)
    # Select *non-terminal* child node
    assert [0b001000000,
            0b000100000] == select(exploration,
                                   mock_tie_breaker,
                                   mock_get_valid_moves,
                                   mock_is_terminal,
                                   { **initial_state,
                                     'moves': [
                                        {
                                            # 'move': 0b000100000,
                                            # 'state': {'board': [0b01010010, 0b000011010],
                                            #           'player_to_move': 1},
                                            'num_rollouts': 1,
                                            'score': 0
                                            # 'moves': []
                                        },
                                        {
                                            'move': 0b001000000,
                                            'state': {'board': [0b01100010, 0b000011010],
                                                      'player_to_move': 1},
                                            'num_rollouts': 2,
                                            'score': 2,
                                            # Make stats of these two moves identical, to
                                            # test that select() intelligently picks the
                                            # non-terminal state.
                                            # (The tie-breaker would otherwise
                                            # *incorrectly* pick the move at index 0)
                                            'moves': [
                                                {
                                                    # 'move': 0b100000000,
                                                    # 'state': {'board': [0b01100010, 0b100011010],
                                                    #           'player_to_move': 0},
                                                    'num_rollouts': 1,
                                                    'score': 0,
                                                    # 'moves': []
                                                },
                                                # This is a terminal state.
                                                # We should not be choosing terminal
                                                # states for selection
                                                {
                                                    'move': 0b000100000,
                                                    'state': {'board': [0b01100010, 0b000111010],
                                                              'player_to_move': 0},
                                                    'num_rollouts': 1,
                                                    'score': 0,
                                                    'moves': []
                                                }
                                            ]
                                        },
                                        {
                                            # 'move': 0b100000000,
                                            # 'state': {'board': [0b11000010, 0b000011010],
                                            #         'player_to_move': 1},
                                            'num_rollouts': 2,
                                            'score': 1
                                            # 'moves': []
                                        }
                                    ]})

    # Test with a tree that causes select() to choose a terminal state (draw)

    def mock_get_valid_moves_2(state): # type: (State) -> list[Move]
        board = state['board']

        if board == [0b101010010, 0b010000101]:
            return [0b000001000, 0b000100000]

        if board == [0b101010010, 0b010001101]:
            return [0b000100000]

        return []

    def mock_is_terminal_2(state): # type: (State) -> bool
        board = state['board']
        return board in [[0b101011010, 0b010100101], [0b101110010, 0b010001101]]

    assert [0b000001000,
            0b000100000] == select(1,
                                   mock_tie_breaker,
                                   mock_get_valid_moves_2,
                                   mock_is_terminal_2,
                                   {
                                        'state': { 'board': [0b101010010, 0b010000101],
                                                   'player_to_move': 1 },
                                        'num_rollouts': 4,
                                        # 'score': 0,
                                        'moves': [{
                                                    # 'state': { 'board': [0b101010010,
                                                    #                      0b010100101],
                                                    #            'player_to_move': 0 },
                                                    # 'move': 0b000100000,
                                                    'num_rollouts': 2,
                                                    'score': 0
                                                    # 'moves': [{
                                                    #          'state': { 'board': [0b101011010,
                                                    #                               0b010100101],
                                                    #                     'player_to_move': 1 },
                                                    #          'move': 0b000001000,
                                                    #          'num_rollouts': 1,
                                                    #          'score': 0,
                                                    #          'moves': []
                                                    #         }]
                                                },
                                                {
                                                    'state': { 'board': [0b101010010, 0b010001101],
                                                               'player_to_move': 0 },
                                                    'move': 0b000001000,
                                                    'num_rollouts': 2,
                                                    'score': 0,
                                                    'moves': [{
                                                            'state': { 'board': [0b101110010,
                                                                                 0b010001101],
                                                                       'player_to_move': 1 },
                                                            'move': 0b000100000,
                                                            'num_rollouts': 1,
                                                            'score': 0,
                                                            'moves': []
                                                            }]
                                                }]
                                   })

    # TO-DO: Write test with a tree that causes select() to choose a terminal state
    # where there is a win BUT the board is NOT empty
    # select() should not drill further down!

def test_treewalk():
    assert {
        'move': 0b000000010,
        # 'state': { 'board': [0b000000001, 0b000000010],
        #            'player_to_move': 0 },
        # 'num_rollouts': 2,
        # 'score': 1,
        'moves': [{'move': 0b000000100,
                 # 'state': { 'board': [0b000000101, 0b000000010],
                 #            'player_to_move': 1 },
                 # 'num_rollouts': 1,
                 # 'score': 1,
                 'moves': []}]
    } == treewalk([0b000000001, 0b000000010],
                  { 'move': 0b000000000,
                    # 'state': { 'board': [0b000000000, 0b000000000],
                    #            'player_to_move': 0 },
                    # 'num_rollouts': 4,
                    # 'score': 3,
                    'moves': [{ 'move': 0b000000001,
                                # 'state': { 'board': [0b000000001, 0b000000000],
                                #            'player_to_move': 1 },
                                # 'num_rollouts': 1,
                                # 'score': 1,
                                'moves': [{ 'move': 0b000000010,
                                            # 'state': { 'board': [0b000000001, 0b000000010],
                                            #            'player_to_move': 0 },
                                            # 'num_rollouts': 2,
                                            # 'score': 1,
                                            'moves': [{ 'move': 0b000000100,
                                                        # 'state': { 'board': [0b000000101,
                                                        #                      0b000000010],
                                                        #            'player_to_move': 1 },
                                                        # 'num_rollouts': 1,
                                                        # 'score': 1,
                                                        'moves': []}]}]}]})

    assert {
        'move': 0b000000000,
        # 'state': { 'board': [0b000000000, 0b000000000],
        #            'player_to_move': 0 },
        # 'num_rollouts': 4,
        # 'score': 3,
        'moves': [{ 'move': 0b000000001,
                 # 'state': { 'board': [0b000000001, 0b000000000],
                 #            'player_to_move': 1 },
                 # 'num_rollouts': 1,
                 # 'score': 1,
                 'moves': []}]
    } == treewalk([],
                  { 'move': 0b000000000,
                   # 'state': { 'board': [0b000000000, 0b000000000],
                   #            'player_to_move': 0 },
                   # 'num_rollouts': 4,
                   # 'score': 3,
                   'moves': [{ 'move': 0b000000001,
                            # 'state': { 'board': [0b000000001, 0b000000000],
                            #            'player_to_move': 1 },
                            # 'num_rollouts': 1,
                            # 'score': 1,
                            'moves': []}]})

def test_replace_node():
    assert {
        'move': 0b000000000,
        # 'state': { 'board': [0b000000000, 0b000000000],
        #            'player_to_move': 0 },
        # 'num_rollouts': 4,
        # 'score': 3,
        'moves': [{ 'move': 0b000000001,
                   # 'state': { 'board': [0b000000001, 0b000000000],
                   #            'player_to_move': 1 },
                   # 'num_rollouts': 1,
                   # 'score': 1,
                   'moves': [{ 'move': 0b000000010,
                              # 'state': { 'board': [0b000000001, 0b000000010],
                              #            'player_to_move': 0 },
                              # 'num_rollouts': 2,
                              # 'score': 1,
                              'moves': [{ 'move': 0b000000100,
                                         # 'state': { 'board': [0b000000101, 0b000000010],
                                         #            'player_to_move': 1 },
                                         # 'num_rollouts': 1,
                                         # 'score': 1,
                                         'moves': [] },
                                        { 'move': 0b000001000,
                                         # 'state': { 'board': [0b000001001, 0b000000010],
                                         #            'player_to_move': 1 },
                                         # 'num_rollouts': 1,
                                         # 'score': 1,
                                         'moves': []}]}]}]
    } == replace_node({ 'move': 0b000000000,
                        # 'state': { 'board': [0b000000000, 0b000000000],
                        #            'player_to_move': 0 },
                        # 'num_rollouts': 4,
                        # 'score': 3,
                        'moves': [{ 'move': 0b000000001,
                                    # 'state': { 'board': [0b000000001, 0b000000000],
                                    #            'player_to_move': 1 },
                                    # 'num_rollouts': 1,
                                    # 'score': 1,
                                    'moves': [{ 'move': 0b000000010,
                                                # 'state': { 'board': [0b000000001, 0b000000010],
                                                #            'player_to_move': 0 },
                                                # 'num_rollouts': 2,
                                                # 'score': 1,
                                                'moves': [{ 'move': 0b000000100,
                                                            # 'state': { 'board': [0b000000101,
                                                            #                      0b000000010],
                                                            #            'player_to_move': 1 },
                                                            # 'num_rollouts': 1,
                                                            # 'score': 1,
                                                            'moves': [] }]}]}]
                      },
                      [0b000000001, 0b000000010],
                      { 'move': 0b000000010,
                        # 'state': { 'board': [0b000000001, 0b000000010],
                        #            'player_to_move': 0 },
                        # 'num_rollouts': 2,
                        # 'score': 1,
                        'moves': [{ 'move': 0b000000100,
                                    # 'state': { 'board': [0b000000101, 0b000000010],
                                    #            'player_to_move': 1 },
                                    # 'num_rollouts': 1,
                                    # 'score': 1,
                                    'moves': [] },
                                  { 'move': 0b000001000,
                                    # 'state': { 'board': [0b000001001, 0b000000010],
                                    #            'player_to_move': 1 },
                                    # 'num_rollouts': 1,
                                    # 'score': 1,
                                    'moves': [] }]
                      })

def test_simulate():
    def mock_is_terminal(state): # type: (State) -> bool
        return reduce(lambda found_match, terminal_state:
                        found_match or state['board'] == terminal_state,
                      [[0b110100101, 0b001011010],
                       [0b011100101, 0b100011010],
                       [0b011000101, 0b000111010],
                       [0b110000101, 0b000111010]],
                      False)

    def mock_check_win(state): # type: (State) -> int | None
        board = state['board']

        return (0    if board == [0b110100101, 0b001011010] else
                None if board == [0b011100101, 0b100011010] else
                1    if board in [[0b110000101, 0b000111010],
                                  [0b011000101, 0b000111010]]
                else -1)

    def mock_valid_moves(state): # type: (State) -> list[Move]
        board = state['board']

        return ([0b001000000, 0b100000000] if board == [0b010100101, 0b000011010] else
                [0b000100000, 0b100000000] if board == [0b011000101, 0b000011010] else
                [0b000100000, 0b001000000] if board == [0b110000101, 0b000011010] else
                [0b100000000]              if board == [0b010100101, 0b001011010] else
                [0b001000000]              if board == [0b010100101, 0b100011010] else
                [0b000100000]              if board == [0b011000101, 0b100011010] else
                [0b000100000]              if board == [0b110000101, 0b001011010]
                else [0x1BADC0DE])

    seed(123)
    def mock_random_int(): # type: () -> int
        return randint(0, maxsize)

    def mock_apply_move_to_state(state, move): # type: (State, Move) -> State
        # 2-ply states
        if state == { 'board': [0b010100101, 0b000011010], 'player_to_move': 1 }:
            if move == 0b001000000:
                return { 'board': [0b010100101, 0b001011010], 'player_to_move': 0 }
            # 0b100000000
            return     { 'board': [0b010100101, 0b100011010], 'player_to_move': 0 }
        if state == { 'board': [0b011000101, 0b000011010], 'player_to_move': 1 }:
            if move == 0b000100000:
                return { 'board': [0b011000101, 0b000111010], 'player_to_move': 0 }
            # 0b100000000
            return     { 'board': [0b011000101, 0b100011010], 'player_to_move': 0 }
        if state == { 'board': [0b110000101, 0b000011010], 'player_to_move': 1 }:
            if move == 0b000100000:
                return { 'board': [0b110000101, 0b000111010], 'player_to_move': 0 }
            # 0b001000000
            return     { 'board': [0b110000101, 0b001011010], 'player_to_move': 0 }
        # 1-ply states
        if state == { 'board': [0b010100101, 0b001011010], 'player_to_move': 0 }:
            return { 'board': [0b110100101, 0b001011010], 'player_to_move': 1 }
        if state == { 'board': [0b010100101, 0b100011010], 'player_to_move': 0 }:
            return { 'board': [0b110100101, 0b001011010], 'player_to_move': 1 }
        if state == { 'board': [0b011000101, 0b100011010], 'player_to_move': 0 }:
            return { 'board': [0b011100101, 0b100011010], 'player_to_move': 1 }
        if state == { 'board': [0b110000101, 0b001011010], 'player_to_move': 0 }:
            return { 'board': [0b110100101, 0b001011010], 'player_to_move': 1 }

    # 2-play moves
    # ------------
    # Win or draw possible
    assert simulate(
        mock_is_terminal,
        mock_check_win,
        mock_valid_moves,
        mock_random_int,
        mock_apply_move_to_state,
        { 'board': [0b010100101, 0b000011010], 'player_to_move': 1 }
    ) in [None, 0]
    # Lose or draw possible
    assert simulate(
        mock_is_terminal,
        mock_check_win,
        mock_valid_moves,
        mock_random_int,
        mock_apply_move_to_state,
        { 'board': [0b011000101, 0b000011010], 'player_to_move': 1 }
    ) in [None, 1]
    # Win or lose possible
    assert simulate(
        mock_is_terminal,
        mock_check_win,
        mock_valid_moves,
        mock_random_int,
        mock_apply_move_to_state,
        { 'board': [0b110000101, 0b000011010], 'player_to_move': 1 }
    ) in [0, 1]

    # Terminal states
    # ---------------
    # Terminal state: win player 1
    assert 0 == simulate(
        mock_is_terminal,
        mock_check_win,
        mock_valid_moves,
        mock_random_int,
        mock_apply_move_to_state,
        { 'board': [0b110100101, 0b001011010], 'player_to_move': 1 }
    )
    # Terminal state: draw
    assert None is simulate(
        mock_is_terminal,
        mock_check_win,
        mock_valid_moves,
        mock_random_int,
        mock_apply_move_to_state,
        { 'board': [0b011100101, 0b100011010], 'player_to_move': 1 }
    )
    # Terminal state: lose (player 2 wins)
    assert 1 == simulate(
        mock_is_terminal,
        mock_check_win,
        mock_valid_moves,
        mock_random_int,
        mock_apply_move_to_state,
        { 'board': [0b011000101, 0b000111010], 'player_to_move': 0 }
    )

def test_is_path_valid():
    tree = { 'move': 0b000001000,
             # 'state': { 'board': [0b010000101, 0b000011010],
             #            'player_to_move': 0 },
             # 'num_rollouts': 4,
             # 'score': 2,
             'moves': [{ 'move': 0b100000000,
                      # 'state': { 'board': [0b110000101, 0b000011010],
                      #            'player_to_move': 1 },
                      # 'num_rollouts': 2,
                      # 'score': 1,
                      'moves': [{ 'move': 0b001000000,
                               # 'state': { 'board': [0b110000101, 0b001011010],
                               #            'player_to_move': 0 },
                               # 'num_rollouts': 0,
                               # 'score': 0,
                               'moves': []}]}]}

    assert True is is_path_valid(tree, [0b100000000, 0b001000000])
    assert False is is_path_valid(tree, [0b100000000, 0b000000001])
    assert False is is_path_valid(tree, [0b100000000, 0b001000000, 0b000000001])
    assert True is is_path_valid(tree, [])

# Not testing if an invalid path is passed in.
# For now it will throw a null pointer exception
def test_backprop():
    initial_state = { 'move': 0b000001000,
                      'state': { 'board': [0b010000101, 0b000011010],
                                 'player_to_move': 0 },
                      'num_rollouts': 4,
                      'score': 2,
                      'moves': [{ 'move': 0b001000000,
                                  'state': { 'board': [0b011000101, 0b000011010],
                                             'player_to_move': 1 },
                                  'num_rollouts': 1,
                                  'score': 0,
                                  'moves': [] },
                                  { 'move': 0b100000000,
                                    'state': { 'board': [0b110000101, 0b000011010],
                                               'player_to_move': 1 },
                                    'num_rollouts': 2,
                                    'score': 1,
                                    'moves': [{ 'move': 0b000100000,
                                                'state': { 'board': [0b110000101, 0b000111010],
                                                           'player_to_move': 0 },
                                                'num_rollouts': 1,
                                                'score': 0,
                                                'moves': [] },
                                              # This is the leaf node to backprop from
                                              { 'move': 0b001000000,
                                                'state': { 'board': [0b110000101, 0b001011010],
                                                           'player_to_move': 0 },
                                                'num_rollouts': 0,
                                                'score': 0,
                                                'moves': []}]}]}

    assert { 'move': 0b000001000,
             'state': { 'board': [0b010000101, 0b000011010],
                        'player_to_move': 0 },
             'num_rollouts': 5,
             'score': 1,
             'moves': [{ 'move': 0b001000000,
                         'state': { 'board': [0b011000101, 0b000011010],
                                    'player_to_move': 1 },
                         'num_rollouts': 1,
                         'score': 0,
                         'moves': [] },
                       { 'move': 0b100000000,
                           'state': { 'board': [0b110000101, 0b000011010],
                                      'player_to_move': 1 },
                           'num_rollouts': 3,
                           'score': 2,
                           'moves': [{ 'move': 0b000100000,
                                       'state': { 'board': [0b110000101, 0b000111010],
                                                  'player_to_move': 0 },
                                       'num_rollouts': 1,
                                       'score': 0,
                                       'moves': [] },
                                     # This is the leaf node to backprop from
                                     { 'move': 0b001000000,
                                       'state': { 'board': [0b110000101, 0b001011010],
                                                  'player_to_move': 0 },
                                       'num_rollouts': 1,
                                       'score': -1,
                                       'moves': []}]}]
    } == backprop(0, [0b100000000, 0b001000000], initial_state, 1)

    assert { 'move': 0b000001000,
             'state': { 'board': [0b010000101, 0b000011010],
                        'player_to_move': 0 },
             'num_rollouts': 5,
             'score': 2,
             'moves': [{ 'move': 0b001000000,
                         'state': { 'board': [0b011000101, 0b000011010],
                                    'player_to_move': 1 },
                         'num_rollouts': 1,
                         'score': 0,
                         'moves': [] },
                       { 'move': 0b100000000,
                           'state': { 'board': [0b110000101, 0b000011010],
                                      'player_to_move': 1 },
                           'num_rollouts': 3,
                           'score': 1,
                           'moves': [{ 'move': 0b000100000,
                                       'state': { 'board': [0b110000101, 0b000111010],
                                                  'player_to_move': 0 },
                                       'num_rollouts': 1,
                                       'score': 0,
                                       'moves': [] },
                                     # This is the leaf node to backprop from
                                     { 'move': 0b001000000,
                                       'state': { 'board': [0b110000101, 0b001011010],
                                                  'player_to_move': 0 },
                                       'num_rollouts': 1,
                                       'score': 0,
                                       'moves': []}]}]
    } == backprop(None, [0b100000000, 0b001000000], initial_state, 1)

    assert { 'move': 0b000001000,
             'state': { 'board': [0b010000101, 0b000011010],
                        'player_to_move': 0 },
             'num_rollouts': 5,
             'score': 3,
             'moves': [{ 'move': 0b001000000,
                         'state': { 'board': [0b011000101, 0b000011010],
                                    'player_to_move': 1 },
                         'num_rollouts': 1,
                         'score': 0,
                         'moves': [] },
                       { 'move': 0b100000000,
                         'state': { 'board': [0b110000101, 0b000011010],
                                    'player_to_move': 1 },
                         'num_rollouts': 3,
                         'score': 0,
                         'moves': [{ 'move': 0b000100000,
                                     'state': { 'board': [0b110000101, 0b000111010],
                                                'player_to_move': 0 },
                                     'num_rollouts': 1,
                                     'score': 0,
                                     'moves': [] },
                                   # This is the leaf node to backprop from
                                   { 'move': 0b001000000,
                                     'state': { 'board': [0b110000101, 0b001011010],
                                                'player_to_move': 0 },
                                     'num_rollouts': 1,
                                     'score': 1,
                                     'moves': []}]}]
    } == backprop(1, [0b100000000, 0b001000000], initial_state, 1)
