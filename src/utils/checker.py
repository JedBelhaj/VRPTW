
from utils.distance import euclidean_by_id


INFEASIBLE_ROUTE = (False, float("inf"), float("inf"), float("inf"))

# Travel time unit = Distance unit
# Route schedule time = travel + wait + service

def clone_routes(routes):
    return [list(route) for route in routes]


def route_start_times(problem, route):
    starts = []
    time = 0.0
    current = route[0]

    for node in route[1:]:
        travel = euclidean_by_id(problem, current, node)
        arrival = time + travel
        customer = problem.customers[node]
        start = max(arrival, customer.ready_time)
        if start > customer.due_time:
            return None
        starts.append(start)
        time = start + customer.service_time
        current = node

    return starts


def evaluate_route(problem, route):
    if len(route) < 2 or route[0] != problem.depot_id or route[-1] != problem.depot_id:
        return INFEASIBLE_ROUTE

    load = 0.0
    time = 0.0
    distance = 0.0
    current = route[0]

    for node in route[1:]:
        travel = euclidean_by_id(problem, current, node)
        arrival = time + travel
        customer = problem.customers[node]
        start = max(arrival, customer.ready_time)

        if start > customer.due_time:
            return INFEASIBLE_ROUTE

        distance += travel
        time = start + customer.service_time
        current = node

    for node in route[1:-1]:
        load += problem.customers[node].demand

    if load > problem.capacity:
        return INFEASIBLE_ROUTE

    return True, distance, load, time


def evaluate_solution(problem, routes):
    if len(routes) > problem.vehicle_count:
        return (
            False,
            float("inf"),
            f"Vehicle limit exceeded: routes={len(routes)}, vehicles={problem.vehicle_count}",
        )

    seen = set()
    total_distance = 0.0

    for route in routes:
        feasible, distance, _, _ = evaluate_route(problem, route)
        if not feasible:
            return False, float("inf"), "Infeasible route"
        total_distance += distance

        for customer_id in route[1:-1]:
            if customer_id in seen:
                return False, float("inf"), f"Customer visited more than once: {customer_id}"
            seen.add(customer_id)

    all_customers = set(problem.customer_ids)
    if seen != all_customers:
        missing = sorted(all_customers - seen)
        return False, float("inf"), f"Missing customers: {missing[:10]}"

    return True, total_distance, "OK"
