"""Route-focused helper functions for Tabu Search."""

from typing import List

from models.problem import ProblemInstance
from utils.checker import evaluate_solution


def clean_routes(problem: ProblemInstance, routes: List[List[int]]) -> List[List[int]]:
    """Keep only non-empty depot-anchored routes."""
    return [
        route
        for route in routes
        if len(route) > 2 and route[0] == problem.depot_id and route[-1] == problem.depot_id
    ]


def total_distance(problem: ProblemInstance, routes: List[List[int]]) -> float:
    """Return solution distance or inf when infeasible."""
    feasible, total, _ = evaluate_solution(problem, routes)
    if not feasible:
        return float("inf")
    return total
