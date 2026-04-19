import random

from utils.checker import evaluate_route


def random_feasible_solution(problem):
    rng = random.Random()
    depot = problem.depot_id
    customers = list(problem.customer_ids)
    rng.shuffle(customers)

    routes = []

    for customer_id in customers:
        inserted = False
        route_indices = list(range(len(routes)))
        rng.shuffle(route_indices)

        for r_idx in route_indices:
            route = routes[r_idx]
            positions = list(range(1, len(route)))
            rng.shuffle(positions)
            for pos in positions:
                candidate = route[:pos] + [customer_id] + route[pos:]
                feasible, _, _, _ = evaluate_route(problem, candidate)
                if feasible:
                    routes[r_idx] = candidate
                    inserted = True
                    break
            if inserted:
                break

        if not inserted:
            singleton = [depot, customer_id, depot]
            feasible, _, _, _ = evaluate_route(problem, singleton)
            if not feasible:
                raise ValueError(f"Customer {customer_id} cannot be served feasibly.")
            routes.append(singleton)

    return routes

