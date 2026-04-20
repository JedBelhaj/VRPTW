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
    total_moves,
    enabled_operators,
    operator_percentages=None,
):
    candidates = []
    generators = operator_generators()

    active_operators = [op for op in enabled_operators if op in generators]
    if not active_operators:
        return candidates

    total_moves = max(1, int(total_moves))
    operator_percentages = operator_percentages or {}

    weights = {}
    for operator in active_operators:
        raw = operator_percentages.get(operator, 0.0)
        try:
            weight = float(raw)
        except (TypeError, ValueError):
            weight = 0.0
        weights[operator] = max(0.0, weight)

    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        for operator in active_operators:
            weights[operator] = 1.0
        total_weight = float(len(active_operators))

    raw_alloc = {
        operator: (total_moves * weights[operator]) / total_weight for operator in active_operators
    }
    budgets = {operator: int(raw_alloc[operator]) for operator in active_operators}
    assigned = sum(budgets.values())
    remaining = max(0, total_moves - assigned)

    if remaining > 0:
        ranked = sorted(
            active_operators,
            key=lambda op: (raw_alloc[op] - budgets[op], weights[op]),
            reverse=True,
        )
        for idx in range(remaining):
            operator = ranked[idx % len(ranked)]
            budgets[operator] += 1

    for operator in active_operators:
        max_moves = budgets.get(operator, 0)
        if max_moves <= 0:
            continue
        candidates.extend(generators[operator](problem, routes, max_moves=max_moves))

    for cand in candidates:
        cand.routes = clean_routes(problem, cand.routes)
        cand.objective = total_distance(problem, cand.routes)

    candidates.sort(key=lambda c: c.objective)
    return [c for c in candidates if c.objective < float("inf")]

