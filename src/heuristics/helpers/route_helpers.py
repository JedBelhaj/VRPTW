from utils.checker import evaluate_solution


def clean_routes(problem, routes):
    return [
        route
        for route in routes
        if len(route) > 2 and route[0] == problem.depot_id and route[-1] == problem.depot_id
    ]


def total_distance(problem, routes):
    feasible, total, _ = evaluate_solution(problem, routes)
    if not feasible:
        return float("inf")
    return total

