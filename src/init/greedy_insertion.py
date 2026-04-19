import random

from utils.checker import evaluate_route
from utils.distance import euclidean_by_id


def greedy_insertion(problem):
    rng = random.Random()
    depot = problem.depot_id
    unserved = set(problem.customer_ids)
    routes = []

    while unserved:
        route = [depot, depot]

        while True:
            current = route[-2]
            best_customers = []
            best_distance = float("inf")

            candidates = list(unserved)
            rng.shuffle(candidates)
            for customer_id in candidates:
                candidate_route = route[:-1] + [customer_id, depot]
                feasible, distance, _, _ = evaluate_route(problem, candidate_route)
                if not feasible:
                    continue

                leg = euclidean_by_id(problem, current, customer_id)
                if leg < best_distance:
                    best_distance = leg
                    best_customers = [customer_id]
                elif leg == best_distance:
                    best_customers.append(customer_id)

            if not best_customers:
                break

            best_customer = rng.choice(best_customers)
            route = route[:-1] + [best_customer, depot]
            unserved.remove(best_customer)

        if len(route) == 2:
            raise ValueError("No feasible insertion found for remaining customers.")

        routes.append(route)

    return routes

