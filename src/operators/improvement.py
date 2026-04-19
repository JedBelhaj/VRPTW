"""Improvement operator functions used by tabu search."""

from heuristics.helpers.route_helpers import clean_routes, total_distance
from utils.checker import clone_routes, evaluate_route


def _best_and_second_insertion_delta(
    problem,
    routes,
    customer_id,
):
    best = None
    second = None

    for route_idx, route in enumerate(routes):
        feasible, base_distance, _, _ = evaluate_route(problem, route)
        if not feasible:
            continue

        for pos in range(1, len(route)):
            candidate = route[:pos] + [customer_id] + route[pos:]
            cand_feasible, cand_distance, _, _ = evaluate_route(problem, candidate)
            if not cand_feasible:
                continue

            delta = cand_distance - base_distance
            insertion = (delta, route_idx, candidate)
            if best is None or delta < best[0]:
                second = best
                best = insertion
            elif second is None or delta < second[0]:
                second = insertion

    return best, second


def destroy_smallest_route_regret_reinsert(
    problem,
    routes,
    regret_k=2,
):
    """Destroy the smallest route and reinsert customers by regret criterion."""
    if len(routes) < 2:
        return routes, False

    smallest_idx = min(range(len(routes)), key=lambda idx: len(routes[idx]) - 2)
    extracted_customers = routes[smallest_idx][1:-1]
    if not extracted_customers:
        return routes, False

    working = clone_routes(routes)
    working.pop(smallest_idx)
    pending = list(extracted_customers)

    while pending:
        best_customer = None
        best_customer_insertion = None
        max_regret = float("-inf")

        for customer_id in pending:
            best, second = _best_and_second_insertion_delta(problem, working, customer_id)
            if best is None:
                continue

            if regret_k <= 1:
                regret = -best[0]
            elif second is None:
                regret = float("inf")
            else:
                regret = second[0] - best[0]

            if regret > max_regret:
                max_regret = regret
                best_customer = customer_id
                best_customer_insertion = best

        if best_customer is None or best_customer_insertion is None:
            return routes, False

        _, route_idx, inserted_route = best_customer_insertion
        working[route_idx] = inserted_route
        pending.remove(best_customer)

    repaired = clean_routes(problem, working)
    if total_distance(problem, repaired) == float("inf"):
        return routes, False
    return repaired, True
