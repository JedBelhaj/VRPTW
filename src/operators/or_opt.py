
from operators.move_types import add_move
from utils.checker import evaluate_route


def generate_or_opt_moves(problem, routes, max_moves=120):
    moves = []

    for src_idx, src_route in enumerate(routes):
        for chain_len in (2, 3):
            if len(src_route) <= chain_len + 2:
                continue

            for start in range(1, len(src_route) - chain_len):
                chain = src_route[start : start + chain_len]
                src_removed = src_route[:start] + src_route[start + chain_len :]
                f_src, d_src, _, _ = evaluate_route(problem, src_removed)
                if not f_src:
                    continue

                for dst_idx, dst_route in enumerate(routes):
                    for dst_pos in range(1, len(dst_route)):
                        if src_idx == dst_idx and start <= dst_pos <= start + chain_len:
                            continue

                        if src_idx == dst_idx:
                            insert_at = dst_pos
                            if dst_pos > start:
                                insert_at -= chain_len

                            candidate = (
                                src_removed[:insert_at]
                                + chain
                                + src_removed[insert_at:]
                            )
                            feasible, distance, _, _ = evaluate_route(problem, candidate)
                            if not feasible:
                                continue
                            updates = [(src_idx, candidate)]
                            objective = distance
                        else:
                            candidate_dst = dst_route[:dst_pos] + chain + dst_route[dst_pos:]
                            f_dst, d_dst, _, _ = evaluate_route(problem, candidate_dst)
                            if not f_dst:
                                continue
                            updates = [(src_idx, src_removed), (dst_idx, candidate_dst)]
                            objective = d_src + d_dst

                        stop = add_move(
                            moves,
                            routes,
                            updates,
                            ("or_opt", tuple(chain), src_idx, dst_idx),
                            objective,
                            max_moves,
                        )
                        if stop:
                            return moves

    return moves

