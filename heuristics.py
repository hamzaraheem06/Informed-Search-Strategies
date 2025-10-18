import math

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def non_admissible(a, b):
    return 1.5 * manhattan(a, b)

def chebyshev(a, b):
    """
    Chebyshev distance: max(dx, dy)
    Admissible for 4-connected grid with cost=1 (but looser than Manhattan).
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def weighted_euclidean(a, b, weight=1.2):
    """
    Weighted Euclidean: scales Euclidean by a factor.
    Use weight >1 for non-admissible (faster but suboptimal).
    """
    return weight * euclidean(a, b)

def zero_heuristic(a, b):
    return 0
