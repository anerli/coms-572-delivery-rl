from enum import Enum, auto

class DeliveryAction(Enum):
    MOVE_UP = auto()
    MOVE_RIGHT = auto()
    MOVE_DOWN = auto()
    MOVE_LEFT = auto()
    GRAB_UP = auto()
    GRAB_RIGHT = auto()
    GRAB_DOWN = auto()
    GRAB_LEFT = auto()
    DROP_UP = auto()
    DROP_RIGHT = auto()
    DROP_DOWN = auto()
    DROP_LEFT = auto()