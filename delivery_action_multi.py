from enum import Enum
import itertools

class SingleDeliveryAction(Enum):
    # Can't use enum.auto bc auto starts at 1, not 0
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    GRAB = 4
    DROP = 5

# Interpret value given num players
def get_actions(multi_action_value, num_players):
    '''
    E.g. 1 Player:
    0: [0]
    1: [1]
    ...
    5: [5]

    2 Players:
    0: [0, 0]
    1: [0, 1]
    2: [0, 2]
    ...
    35: [5, 5]
    '''
    actions_tup = list(itertools.product([0,1,2,3,4,5], repeat=num_players))[multi_action_value]
    action_enum_list = []
    for action_value in actions_tup:
        action_enum_list.append(SingleDeliveryAction(action_value))
    return action_enum_list

if __name__ == '__main__':
    print(get_actions(34, 2))
    print(get_actions(34, 3))