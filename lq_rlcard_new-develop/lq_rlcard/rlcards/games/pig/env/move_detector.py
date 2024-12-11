from .utils import *


# return the type of the move
def get_move_type(move):
    move_size = len(move)
    if move_size == 0:
        return {'type': TYPE_0_PASS}
    return {'type': TYPE_1_SINGLE, 'suit': move[0] // 100}

    # return {'type': TYPE_15_WRONG}
