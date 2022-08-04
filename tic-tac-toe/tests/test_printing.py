from printing import *

def test_square_owner():
    assert 0 == square_owner(0, [0b000000101, 0b000000110])
    assert 1 == square_owner(1, [0b000000101, 0b000000110])
    assert 0 == square_owner(2, [0b000000101, 0b000000110])
    assert -1 == square_owner(3, [0b000000101, 0b000000110])

def test_game_state_to_string():
    assert 'XOXOX-XOO' == game_state_to_string(9, ['O', 'X'], [0b110001010, 0b001010101])

def test_create_dash_string():
    assert '-' == create_dash_string(1)
    assert '--' == create_dash_string(2)
    assert '---' == create_dash_string(3)

def test_create_row_separator():
    assert '----' == create_row_separator(1, 4)
    assert '---+---' == create_row_separator(2, 3)
    assert '--+--+--' == create_row_separator(3, 2)
    assert '-+-+-+-' == create_row_separator(4, 1)

