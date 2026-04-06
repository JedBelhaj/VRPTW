from typing import List

from models.problem import ProblemInstance
from utils.checker import evaluate_route


def greedy_insertion(problem: ProblemInstance) -> List[List[int]]:
    """Build routes by repeatedly appending the nearest feasible unserved customer."""
    depot = problem.depot_id
    unserved = set(problem.customer_ids)
    routes: List[List[int]] = []

    while unserved:
        route = [depot, depot]

        while True:
            current = route[-2]
            best_customer = None
            best_distance = float("inf")

            for customer_id in unserved:
                candidate_route = route[:-1] + [customer_id, depot]
                feasible, distance, _, _ = evaluate_route(problem, candidate_route)
                if not feasible:
                    continue

                leg = ((problem.customers[current].x - problem.customers[customer_id].x) ** 2 + (problem.customers[current].y - problem.customers[customer_id].y) ** 2) ** 0.5
                if leg < best_distance:
                    best_distance = leg
                    best_customer = customer_id

            if best_customer is None:
                break

            route = route[:-1] + [best_customer, depot]
            unserved.remove(best_customer)

        if len(route) == 2:
            raise ValueError("No feasible insertion found for remaining customers.")

        routes.append(route)

    return routes
