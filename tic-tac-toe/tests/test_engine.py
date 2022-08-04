from engine import *


def test_separate_bitboard():
    assert[0b000000010,
           0b000010000,
           0b000100000,
           0b010000000] == separate_bitboard(0b010110010)

def test_get_valid_moves_bitmask():
    assert 0 == get_valid_moves_bitmask([0b000011111, 0b111100000])
    assert 0b000000011 == get_valid_moves_bitmask([0b000111100, 0b111000000])

def test_apply_move():
    assert [1, 0] == apply_move([0, 0], 0, 1)
    assert [0, 1] == apply_move([0, 0], 1, 1)
    # Should we reject the entire move, or only the out of bounds moves in the bitmask?
    # Only rejecting the out of bounds moves for now
    assert [0, 1] == apply_move([0, 0], 1, 0b1000000001)
    assert [0, 1] == apply_move([0, 1], 0, 1)

def test_apply_move_to_state():
    assert {'board': [0b100010010, 0b011000101],
            'current_player': 0} == apply_move_to_state({'board': [0b100010010, 0b011000100],
                                                         'current_player': 1},
                                                        0b000000001)
    assert {'board': [0b100011010, 0b011000101],
            'current_player': 1} == apply_move_to_state({'board': [0b100010010, 0b011000101],
                                                         'current_player': 0},
                                                        0b000001000)

def test_is_full():
    assert False == is_full([0b001001001, 0])
    assert True == is_full([0b001001001, 0b110110110])

def test_check_win():
    assert 0 == check_win([0b001001001, 0])
    assert 1 == check_win([0, 0b111000000])
    assert None == check_win([0b000001001, 0b000000110])

def test_is_terminal():
    # Win for player 1
    assert True == is_terminal([0b001001101, 0b100010010])
    # Draw: full board
    assert True == is_terminal([0b001110011, 0b110001100])
    # Non-terminal
    assert False == is_terminal([1, 0])

def test_get_valid_moves_list():
    assert None == get_valid_moves_list([0b000011111, 0b111100000])
    assert [0b000000010, 0b000000100] == get_valid_moves_list([0b010110001, 0b101001000])
    # Terminal states should return None
    assert None == get_valid_moves_list([0b001100001, 0b010010010])
