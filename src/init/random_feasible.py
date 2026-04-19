import random

from models.problem import ProblemInstance
from utils.checker import evaluate_route


def random_feasible_solution(problem, seed=0):
    rng = random.Random(seed)
    depot = problem.depot_id
    customers = list(problem.customer_ids)
    rng.shuffle(customers)

    routes = []

    for customer_id in customers:
        inserted = False

        for r_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                candidate = route[:pos] + [customer_id] + route[pos:]
                feasible, _, _, _ = evaluate_route(problem, candidate)
                if feasible:
                    routes[r_idx] = candidate
                    inserted = True
                    break
            if inserted:
                break

        if inserted:
            continue

        singleton = [depot, customer_id, depot]
        feasible, _, _, _ = evaluate_route(problem, singleton)
        if not feasible:
            raise ValueError(f"Customer {customer_id} cannot be served feasibly.")
        routes.append(singleton)

    return routes

