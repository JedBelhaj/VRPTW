from typing import Dict, List, Tuple

from models.problem import ProblemInstance
from utils.checker import evaluate_route


def _distance(problem: ProblemInstance, i: int, j: int) -> float:
    c1 = problem.customers[i]
    c2 = problem.customers[j]
    dx = c1.x - c2.x
    dy = c1.y - c2.y
    return (dx * dx + dy * dy) ** 0.5


def _find_route_index(routes: List[List[int]], customer_id: int) -> int:
    for idx, route in enumerate(routes):
        if customer_id in route[1:-1]:
            return idx
    return -1


def clarke_wright_savings(problem: ProblemInstance) -> List[List[int]]:
    depot = problem.depot_id
    routes = [[depot, cid, depot] for cid in problem.customer_ids]

    savings: List[Tuple[float, int, int]] = []
    customers = problem.customer_ids
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            saving = _distance(problem, depot, ci) + _distance(problem, depot, cj) - _distance(problem, ci, cj)
            savings.append((saving, ci, cj))

    savings.sort(reverse=True, key=lambda x: x[0])

    for _, i, j in savings:
        ri = _find_route_index(routes, i)
        rj = _find_route_index(routes, j)
        if ri == -1 or rj == -1 or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        merged = None
        if route_i[-2] == i and route_j[1] == j:
            merged = route_i[:-1] + route_j[1:]
        elif route_j[-2] == j and route_i[1] == i:
            merged = route_j[:-1] + route_i[1:]

        if merged is None:
            continue

        feasible, _, _, _ = evaluate_route(problem, merged)
        if not feasible:
            continue

        for idx in sorted([ri, rj], reverse=True):
            routes.pop(idx)
        routes.append(merged)

    return routes
