from operators.cross_exchange import generate_cross_exchange_moves
from operators.or_opt import generate_or_opt_moves
from operators.relocate import generate_relocation_moves
from operators.swap import generate_swap_moves
from operators.two_opt import generate_two_opt_inter_moves, generate_two_opt_intra_moves

from .route_helpers import clean_routes, total_distance


def operator_generators():
    return {
        "relocate": generate_relocation_moves,
        "swap": generate_swap_moves,
        "two_opt_intra": generate_two_opt_intra_moves,
        "two_opt_inter": generate_two_opt_inter_moves,
        "or_opt": generate_or_opt_moves,
        "cross_exchange": generate_cross_exchange_moves,
    }


def is_inter_route_move(move_key):
    if not move_key:
        return False

    operator = move_key[0]
    if operator in ("swap", "two_opt_inter", "cross_exchange"):
        return True
    if operator in ("relocate", "or_opt") and len(move_key) >= 4:
        return move_key[2] != move_key[3]
    return False


def generate_candidates(
    problem,
    routes,
    per_operator,
    enabled_operators,
):
    candidates = []
    generators = operator_generators()

    for operator in enabled_operators:
        if operator not in generators:
            continue
        candidates.extend(generators[operator](problem, routes, max_moves=per_operator))

    for cand in candidates:
        cand.routes = clean_routes(problem, cand.routes)
        cand.objective = total_distance(problem, cand.routes)

    candidates.sort(key=lambda c: c.objective)
    return [c for c in candidates if c.objective < float("inf")]

