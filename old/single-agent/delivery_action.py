from enum import Enum

class DeliveryAction(Enum):
    # Can't use enum.auto bc auto starts at 1, not 0
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    GRAB = 4
    DROP = 5