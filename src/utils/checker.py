from typing import Dict, List, Optional, Tuple

from models.problem import ProblemInstance


def euclidean(problem: ProblemInstance, c1_id: int, c2_id: int) -> float:
    c1 = problem.customers[c1_id]
    c2 = problem.customers[c2_id]
    dx = c1.x - c2.x
    dy = c1.y - c2.y
    return (dx * dx + dy * dy) ** 0.5


def clone_routes(routes: List[List[int]]) -> List[List[int]]:
    return [list(route) for route in routes]


def route_load(problem: ProblemInstance, route: List[int]) -> float:
    return sum(problem.customers[cid].demand for cid in route[1:-1])


def route_start_times(problem: ProblemInstance, route: List[int]) -> Optional[List[float]]:
    starts: List[float] = []
    time = 0.0
    current = route[0]

    for node in route[1:]:
        travel = euclidean(problem, current, node)
        arrival = time + travel
        customer = problem.customers[node]
        start = max(arrival, customer.ready_time)
        if start > customer.due_time:
            return None
        starts.append(start)
        time = start + customer.service_time
        current = node

    return starts


def evaluate_route(problem: ProblemInstance, route: List[int]) -> Tuple[bool, float, float, float]:
    if len(route) < 2 or route[0] != problem.depot_id or route[-1] != problem.depot_id:
        return False, float("inf"), float("inf"), float("inf")

    load = 0.0
    time = 0.0
    distance = 0.0
    current = route[0]

    for node in route[1:]:
        travel = euclidean(problem, current, node)
        arrival = time + travel
        customer = problem.customers[node]
        start = max(arrival, customer.ready_time)

        if start > customer.due_time:
            return False, float("inf"), float("inf"), float("inf")

        distance += travel
        time = start + customer.service_time
        current = node

    for node in route[1:-1]:
        load += problem.customers[node].demand

    if load > problem.capacity:
        return False, float("inf"), float("inf"), float("inf")

    return True, distance, load, time


def evaluate_solution(problem: ProblemInstance, routes: List[List[int]]) -> Tuple[bool, float, str]:
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


def _best_feasible_insertion(
    problem: ProblemInstance,
    routes: List[List[int]],
    customer_id: int,
) -> Tuple[Optional[int], Optional[List[int]], float]:
    best_route_idx = None
    best_route = None
    best_delta = float("inf")

    for route_idx, route in enumerate(routes):
        base_feasible, base_distance, _, _ = evaluate_route(problem, route)
        if not base_feasible:
            continue

        for pos in range(1, len(route)):
            candidate = route[:pos] + [customer_id] + route[pos:]
            feasible, distance, _, _ = evaluate_route(problem, candidate)
            if not feasible:
                continue

            delta = distance - base_distance
            if delta < best_delta:
                best_delta = delta
                best_route_idx = route_idx
                best_route = candidate

    return best_route_idx, best_route, best_delta


def repair_to_vehicle_limit(
    problem: ProblemInstance,
    routes: List[List[int]],
    target_vehicle_count: Optional[int] = None,
) -> Optional[List[List[int]]]:
    """Try to reduce route count by reinserting customers of one route into others.

    Returns repaired routes if successful, else None.
    """
    target = problem.vehicle_count if target_vehicle_count is None else target_vehicle_count
    working = clone_routes(routes)

    while len(working) > target:
        merged_one = False
        victim_order = sorted(range(len(working)), key=lambda idx: len(working[idx]) - 2)

        for victim_idx in victim_order:
            victim_customers = working[victim_idx][1:-1]
            if not victim_customers:
                continue

            candidate_routes = clone_routes(working)
            candidate_routes.pop(victim_idx)

            success = True
            for customer_id in victim_customers:
                dst_idx, best_route, _ = _best_feasible_insertion(problem, candidate_routes, customer_id)
                if dst_idx is None or best_route is None:
                    success = False
                    break
                candidate_routes[dst_idx] = best_route

            if success:
                working = candidate_routes
                merged_one = True
                break

        if not merged_one:
            return None

    return working


def maybe_repair_to_vehicle_limit(
    problem: ProblemInstance,
    routes: List[List[int]],
    apply_fleet_repair: bool = True,
) -> Tuple[List[List[int]], str]:
    if not apply_fleet_repair or len(routes) <= problem.vehicle_count:
        return routes, ""

    repaired = repair_to_vehicle_limit(problem, routes, target_vehicle_count=problem.vehicle_count)
    if repaired is not None:
        return repaired, " | fleet repair: success"

    return routes, " | fleet repair: failed"


def best_insertion_position(problem: ProblemInstance, route: List[int], customer_id: int) -> Tuple[Optional[int], float]:
    best_position = None
    best_cost = float("inf")

    for pos in range(1, len(route)):
        candidate = route[:pos] + [customer_id] + route[pos:]
        feasible, distance, _, _ = evaluate_route(problem, candidate)
        if feasible and distance < best_cost:
            best_position = pos
            best_cost = distance

    return best_position, best_cost