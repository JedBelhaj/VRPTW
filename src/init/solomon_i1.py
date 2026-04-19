from utils.checker import evaluate_route, route_start_times
from utils.distance import euclidean_by_id


def _best_position_cost(
    problem,
    route,
    customer_id,
    alpha1,
    alpha2,
):
    base_starts = route_start_times(problem, route)
    if base_starts is None:
        return -1, float("inf")

    best_pos = -1
    best_c1 = float("inf")

    for pos in range(1, len(route)):
        i = route[pos - 1]
        j = route[pos]

        candidate = route[:pos] + [customer_id] + route[pos:]
        feasible, _, _, _ = evaluate_route(problem, candidate)
        if not feasible:
            continue

        candidate_starts = route_start_times(problem, candidate)
        if candidate_starts is None:
            continue

        delta_d = (
            euclidean_by_id(problem, i, customer_id)
            + euclidean_by_id(problem, customer_id, j)
            - euclidean_by_id(problem, i, j)
        )

        delta_t = 0.0
        if pos < len(route):
            before_j = base_starts[pos - 1]
            after_j = candidate_starts[pos]
            delta_t = max(0.0, after_j - before_j)

        c1 = alpha1 * delta_d + alpha2 * delta_t
        if c1 < best_c1:
            best_c1 = c1
            best_pos = pos

    return best_pos, best_c1


def solomon_i1(
    problem,
    alpha1=1.0,
    alpha2=0.0,
    lam=1.0,
):
    unserved = sorted(problem.customer_ids, key=lambda cid: problem.customers[cid].due_time)
    routes = []
    depot = problem.depot_id

    while unserved:
        seed = unserved.pop(0)
        route = [depot, seed, depot]
        feasible, _, _, _ = evaluate_route(problem, route)
        if not feasible:
            raise ValueError(f"Seed customer {seed} is infeasible as single route.")

        while True:
            best_customer = None
            best_pos = -1
            best_c2 = float("-inf")

            for customer_id in unserved:
                pos, c1 = _best_position_cost(problem, route, customer_id, alpha1, alpha2)
                if pos == -1:
                    continue

                c2 = lam * euclidean_by_id(problem, depot, customer_id) - c1
                if c2 > best_c2:
                    best_c2 = c2
                    best_customer = customer_id
                    best_pos = pos

            if best_customer is None:
                break

            route = route[:best_pos] + [best_customer] + route[best_pos:]
            unserved.remove(best_customer)

        routes.append(route)

    return routes

