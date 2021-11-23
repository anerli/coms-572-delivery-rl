from enum import Enum

class DeliveryAction(Enum):
    # Can't use enum.auto bc auto starts at 1, not 0
    # MOVE_UP = 0
    # MOVE_RIGHT = 1
    # MOVE_DOWN = 2
    # MOVE_LEFT = 3
    # GRAB_UP = 4
    # GRAB_RIGHT = 5
    # GRAB_DOWN = 6
    # GRAB_LEFT = 7
    # DROP_UP = 8
    # DROP_RIGHT = 9
    # DROP_DOWN = 10
    # DROP_LEFT = 11
    TURN_RIGHT = 0
    TURN_LEFT = 1
    GRAB = 2
    DROP = 3