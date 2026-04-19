
from init.clarke_wright import clarke_wright_savings
from init.greedy_insertion import greedy_insertion
from init.random_feasible import random_feasible_solution
from init.solomon_i1 import solomon_i1
from init.sweep import sweep_algorithm


def get_initial_methods(seed=0):
    return {
        "greedy": greedy_insertion,
        "solomon": solomon_i1,
        "clarke_wright": clarke_wright_savings,
        "random": lambda p: random_feasible_solution(p, seed=seed),
        "sweep": sweep_algorithm,
    }

