from typing import List

from models.problem import ProblemInstance
from operators.move_types import MoveCandidate
from utils.checker import clone_routes, evaluate_route


def generate_swap_moves(problem: ProblemInstance, routes: List[List[int]], max_moves: int = 120) -> List[MoveCandidate]:
    moves: List[MoveCandidate] = []

    for r1 in range(len(routes)):
        for r2 in range(r1 + 1, len(routes)):
            route1 = routes[r1]
            route2 = routes[r2]

            for p1 in range(1, len(route1) - 1):
                for p2 in range(1, len(route2) - 1):
                    c1 = route1[p1]
                    c2 = route2[p2]

                    cand1 = route1[:]
                    cand2 = route2[:]
                    cand1[p1] = c2
                    cand2[p2] = c1

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
                            move_key=("swap", c1, c2, r1, r2),
                            objective=d1 + d2,
                        )
                    )
                    if len(moves) >= max_moves:
                        return moves

    return moves
