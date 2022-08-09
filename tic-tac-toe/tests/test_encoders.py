import numpy as np
from encoders import one_d_to_2_d, state_to_one_plane_encoding


def test_one_d_to_2_d():
    assert (0, 1) == one_d_to_2_d(1, 3)
    assert (1, 1) == one_d_to_2_d(4, 3)
    assert (2, 2) == one_d_to_2_d(8, 3)

def test_state_to_one_plane_encoding():
    result = np.array(
        [[0,  0, 0],
         [0, -1, 0],
         [0,  0, 1]]
    ) == state_to_one_plane_encoding({'board': [0b100000000, 0b000010000],
                                      'player_to_move': 0})

    assert result.all()

    result = np.array(
        [[0,  0, 0],
         [0, -1, 0],
         [0,  0, 1]]
    ) == state_to_one_plane_encoding({'board': [0b000010000, 0b100000000],
                                      'player_to_move': 1})

    assert result.all()
