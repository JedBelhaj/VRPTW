from typing import List

from models.problem import ProblemInstance
from operators.move_types import MoveCandidate
from utils.checker import clone_routes, evaluate_route


def generate_or_opt_moves(problem: ProblemInstance, routes: List[List[int]], max_moves: int = 120) -> List[MoveCandidate]:
    moves: List[MoveCandidate] = []

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
                        if src_idx == dst_idx and dst_pos >= start and dst_pos <= start + chain_len:
                            continue

                        if src_idx == dst_idx:
                            pos = dst_pos
                            if dst_pos > start:
                                pos -= chain_len
                            candidate = src_removed[:pos] + chain + src_removed[pos:]
                            feasible, distance, _, _ = evaluate_route(problem, candidate)
                            if not feasible:
                                continue
                            new_routes = clone_routes(routes)
                            new_routes[src_idx] = candidate
                            objective = distance
                        else:
                            candidate_dst = dst_route[:dst_pos] + chain + dst_route[dst_pos:]
                            f_dst, d_dst, _, _ = evaluate_route(problem, candidate_dst)
                            if not f_dst:
                                continue
                            new_routes = clone_routes(routes)
                            new_routes[src_idx] = src_removed
                            new_routes[dst_idx] = candidate_dst
                            objective = d_src + d_dst

                        moves.append(
                            MoveCandidate(
                                routes=new_routes,
                                move_key=("or_opt", tuple(chain), src_idx, dst_idx),
                                objective=objective,
                            )
                        )
                        if len(moves) >= max_moves:
                            return moves

    return moves
