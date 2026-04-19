import math

from utils.checker import evaluate_route
from utils.distance import euclidean_by_id


def _angle(problem, customer_id):
    depot = problem.customers[problem.depot_id]
    customer = problem.customers[customer_id]
    return math.atan2(customer.y - depot.y, customer.x - depot.x)


def sweep_algorithm(problem):
    depot = problem.depot_id
    ordered = sorted(problem.customer_ids, key=lambda cid: _angle(problem, cid))

    routes = []
    current_cluster = []
    current_load = 0

    for customer_id in ordered:
        demand = problem.customers[customer_id].demand
        if current_cluster and current_load + demand > problem.capacity:
            routes.extend(_build_cluster_routes(problem, current_cluster))
            current_cluster = []
            current_load = 0

        current_cluster.append(customer_id)
        current_load += demand

    if current_cluster:
        routes.extend(_build_cluster_routes(problem, current_cluster))

    return routes


def _build_cluster_routes(problem, cluster):
    depot = problem.depot_id
    routes = []
    unserved = set(cluster)

    while unserved:
        route = [depot, depot]
        while True:
            best = None
            best_distance = float("inf")
            current = route[-2]

            for customer_id in list(unserved):
                candidate = route[:-1] + [customer_id, depot]
                feasible, _, _, _ = evaluate_route(problem, candidate)
                if not feasible:
                    continue

                d = euclidean_by_id(problem, current, customer_id)
                if d < best_distance:
                    best_distance = d
                    best = customer_id

            if best is None:
                break

            route = route[:-1] + [best, depot]
            unserved.remove(best)

        if len(route) == 2:
            customer_id = unserved.pop()
            singleton = [depot, customer_id, depot]
            feasible, _, _, _ = evaluate_route(problem, singleton)
            if not feasible:
                raise ValueError(f"Customer {customer_id} cannot be served feasibly.")
            routes.append(singleton)
        else:
            routes.append(route)

    return routes

