import random

from utils.checker import evaluate_route
from utils.distance import euclidean_by_id


def _find_route_index(routes, customer_id):
    for idx, route in enumerate(routes):
        if customer_id in route[1:-1]:
            return idx
    return -1


def clarke_wright_savings(problem):
    rng = random.Random()
    depot = problem.depot_id
    routes = [[depot, cid, depot] for cid in problem.customer_ids]

    savings = []
    customers = list(problem.customer_ids)
    rng.shuffle(customers)
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            saving = euclidean_by_id(problem, depot, ci) + euclidean_by_id(problem, depot, cj) - euclidean_by_id(problem, ci, cj)
            savings.append((saving, ci, cj))

    rng.shuffle(savings)
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

