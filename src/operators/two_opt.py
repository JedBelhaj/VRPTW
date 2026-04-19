
from operators.move_types import add_move
from utils.checker import evaluate_route


def generate_two_opt_intra_moves(problem, routes, max_moves=120):
    moves = []

    for route_idx, route in enumerate(routes):
        if len(route) <= 4:
            continue

        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                candidate = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                feasible, dist, _, _ = evaluate_route(problem, candidate)
                if not feasible:
                    continue

                stop = add_move(
                    moves,
                    routes,
                    [(route_idx, candidate)],
                    ("two_opt_intra", route_idx, i, j),
                    dist,
                    max_moves,
                )
                if stop:
                    return moves

    return moves


def generate_two_opt_inter_moves(problem, routes, max_moves=120):
    moves = []

    for r1 in range(len(routes)):
        for r2 in range(r1 + 1, len(routes)):
            route1 = routes[r1]
            route2 = routes[r2]

            for i in range(1, len(route1) - 1):
                for j in range(1, len(route2) - 1):
                    cand1 = route1[:i] + route2[j:]
                    cand2 = route2[:j] + route1[i:]

                    f1, d1, _, _ = evaluate_route(problem, cand1)
                    if not f1:
                        continue
                    f2, d2, _, _ = evaluate_route(problem, cand2)
                    if not f2:
                        continue

                    stop = add_move(
                        moves,
                        routes,
                        [(r1, cand1), (r2, cand2)],
                        ("two_opt_inter", r1, r2, i, j),
                        d1 + d2,
                        max_moves,
                    )
                    if stop:
                        return moves

    return moves

