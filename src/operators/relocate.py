
from models.problem import ProblemInstance
from operators.move_types import MoveCandidate
from utils.checker import clone_routes, evaluate_route


def generate_relocation_moves(problem, routes, max_moves=120):
    moves = []

    for src_idx, src_route in enumerate(routes):
        for src_pos in range(1, len(src_route) - 1):
            customer_id = src_route[src_pos]
            src_removed = src_route[:src_pos] + src_route[src_pos + 1 :]
            src_feasible, src_distance, _, _ = evaluate_route(problem, src_removed)
            if not src_feasible:
                continue

            for dst_idx, dst_route in enumerate(routes):
                for dst_pos in range(1, len(dst_route)):
                    if src_idx == dst_idx and (dst_pos == src_pos or dst_pos == src_pos + 1):
                        continue

                    if src_idx == dst_idx:
                        candidate_route = src_removed[:dst_pos - (1 if dst_pos > src_pos else 0)] + [customer_id] + src_removed[dst_pos - (1 if dst_pos > src_pos else 0) :]
                        feasible, distance, _, _ = evaluate_route(problem, candidate_route)
                        if not feasible:
                            continue
                        new_routes = clone_routes(routes)
                        new_routes[src_idx] = candidate_route
                        objective = distance
                    else:
                        candidate_dst = dst_route[:dst_pos] + [customer_id] + dst_route[dst_pos:]
                        dst_feasible, dst_distance, _, _ = evaluate_route(problem, candidate_dst)
                        if not dst_feasible:
                            continue
                        new_routes = clone_routes(routes)
                        new_routes[src_idx] = src_removed
                        new_routes[dst_idx] = candidate_dst
                        objective = src_distance + dst_distance

                    moves.append(
                        MoveCandidate(
                            routes=new_routes,
                            move_key=("relocate", customer_id, src_idx, dst_idx),
                            objective=objective,
                        )
                    )
                    if len(moves) >= max_moves:
                        return moves

    return moves

