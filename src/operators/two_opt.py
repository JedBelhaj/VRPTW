from typing import List

from models.problem import ProblemInstance
from operators.move_types import MoveCandidate
from utils.checker import clone_routes, evaluate_route


def generate_two_opt_intra_moves(problem: ProblemInstance, routes: List[List[int]], max_moves: int = 120) -> List[MoveCandidate]:
    moves: List[MoveCandidate] = []

    for route_idx, route in enumerate(routes):
        if len(route) <= 4:
            continue

        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                candidate = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
                feasible, dist, _, _ = evaluate_route(problem, candidate)
                if not feasible:
                    continue

                new_routes = clone_routes(routes)
                new_routes[route_idx] = candidate
                moves.append(
                    MoveCandidate(
                        routes=new_routes,
                        move_key=("two_opt_intra", route_idx, i, j),
                        objective=dist,
                    )
                )
                if len(moves) >= max_moves:
                    return moves

    return moves


def generate_two_opt_inter_moves(problem: ProblemInstance, routes: List[List[int]], max_moves: int = 120) -> List[MoveCandidate]:
    moves: List[MoveCandidate] = []

    for r1 in range(len(routes)):
        for r2 in range(r1 + 1, len(routes)):
            route1 = routes[r1]
            route2 = routes[r2]

            for i in range(1, len(route1) - 1):
                for j in range(1, len(route2) - 1):
                    cand1 = route1[:i] + route2[j:]
                    cand2 = route2[:j] + route1[i:]

                    if cand1[-1] != problem.depot_id:
                        cand1.append(problem.depot_id)
                    if cand2[-1] != problem.depot_id:
                        cand2.append(problem.depot_id)

                    f1, d1, _, _ = evaluate_route(problem, cand1)
                    if not f1:
                        continue
                    f2, d2, _, _ = evaluate_route(problem, cand2)
                    if not f2:
                        continue

                    new_routes = clone_routes(routes)
                    new_routes[r1] = cand1
                    new_routes[r2] = cand2
                    moves.append(
                        MoveCandidate(
                            routes=new_routes,
                            move_key=("two_opt_inter", r1, r2, i, j),
                            objective=d1 + d2,
                        )
                    )
                    if len(moves) >= max_moves:
                        return moves

    return moves
