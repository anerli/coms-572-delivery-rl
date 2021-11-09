def manhattan_dist(pos1, pos2):
    return sum(abs(e1-e2) for e1, e2 in zip(pos1, pos2))