
from operators.move_types import add_move
from utils.checker import evaluate_route


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
                        insert_at = dst_pos
                        if dst_pos > src_pos:
                            insert_at -= 1

                        candidate_route = (
                            src_removed[:insert_at]
                            + [customer_id]
                            + src_removed[insert_at:]
                        )
                        feasible, distance, _, _ = evaluate_route(problem, candidate_route)
                        if not feasible:
                            continue
                        updates = [(src_idx, candidate_route)]
                        objective = distance
                    else:
                        candidate_dst = dst_route[:dst_pos] + [customer_id] + dst_route[dst_pos:]
                        dst_feasible, dst_distance, _, _ = evaluate_route(problem, candidate_dst)
                        if not dst_feasible:
                            continue
                        updates = [(src_idx, src_removed), (dst_idx, candidate_dst)]
                        objective = src_distance + dst_distance

                    stop = add_move(
                        moves,
                        routes,
                        updates,
                        ("relocate", customer_id, src_idx, dst_idx),
                        objective,
                        max_moves,
                    )
                    if stop:
                        return moves

    return moves

