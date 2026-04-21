from operators.move_types import add_move
from utils.checker import evaluate_route


def _canonical_edge(a, b):
    return (min(a, b), max(a, b))


def _canonical_edge_pair(edge1, edge2):
    return tuple(sorted((edge1, edge2)))


def _is_degenerate(route):
    """A route is degenerate if it has no customers (just depot-depot)."""
    return len(route) <= 2


def _same_route_set(cand1, cand2, route1, route2):
    return {tuple(cand1), tuple(cand2)} == {tuple(route1), tuple(route2)}


def _is_reversal_of_original(cand1, cand2, route1, route2):
    revs = (tuple(reversed(route1)), tuple(reversed(route2)))
    return tuple(cand1) in revs or tuple(cand2) in revs


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

                edge_pair = _canonical_edge_pair(
                    _canonical_edge(route[i - 1], route[i]),
                    _canonical_edge(route[j], route[j + 1])
                )

                if add_move(
                    moves, routes, [(route_idx, candidate)],
                    ("two_opt_intra", edge_pair, route_idx, i, j),
                    dist, max_moves
                ):
                    return moves

    return moves


def generate_two_opt_inter_moves(problem, routes, max_moves=120):
    moves = []

    for r1 in range(len(routes)):
        route1 = routes[r1]
        if _is_degenerate(route1):
            continue

        for r2 in range(r1 + 1, len(routes)):
            route2 = routes[r2]
            if _is_degenerate(route2):
                continue

            for i in range(1, len(route1) - 1):
                for j in range(1, len(route2) - 1):
                    edge_pair = _canonical_edge_pair(
                        _canonical_edge(route1[i - 1], route1[i]),
                        _canonical_edge(route2[j - 1], route2[j])
                    )

                    depot = route1[0]
                    
                    # Define both variants: (cand1, cand2, variant_name, requires_reversal_check)
                    variants = [
                        (
                            route1[:i] + route2[j:],
                            route2[:j] + route1[i:],
                            "A", False
                        ),
                        (
                            route1[:i] + list(reversed(route2[1:j])) + [depot],
                            [depot] + list(reversed(route1[i:-1])) + route2[j:],
                            "B", True
                        )
                    ]

                    # Loop through both variants using a single evaluation block
                    for c1, c2, var_name, check_rev in variants:
                        if _is_degenerate(c1) or _is_degenerate(c2):
                            continue
                        if _same_route_set(c1, c2, route1, route2):
                            continue
                        if check_rev and _is_reversal_of_original(c1, c2, route1, route2):
                            continue

                        # Evaluate Route 1
                        f1, d1, _, _ = evaluate_route(problem, c1)
                        if not f1:
                            continue
                            
                        # Evaluate Route 2
                        f2, d2, _, _ = evaluate_route(problem, c2)
                        if not f2:
                            continue

                        # Add move if both are feasible
                        if add_move(
                            moves, routes, [(r1, c1), (r2, c2)],
                            ("two_opt_inter", edge_pair, r1, r2, i, j, var_name),
                            d1 + d2, max_moves
                        ):
                            return moves

    return moves