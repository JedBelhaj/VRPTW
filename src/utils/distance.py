import math


def euclidean(c1, c2):
    return math.hypot(c1.x - c2.x, c1.y - c2.y)


def euclidean_by_id(problem, c1_id, c2_id):
    c1 = problem.customers[c1_id]
    c2 = problem.customers[c2_id]
    return math.hypot(c1.x - c2.x, c1.y - c2.y)