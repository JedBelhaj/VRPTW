import random
from typing import Callable, Dict, List, Optional, Tuple

from init.methods import get_initial_methods
from models.problem import ProblemInstance
from operators.cross_exchange import generate_cross_exchange_moves
from operators.move_types import MoveCandidate
from operators.or_opt import generate_or_opt_moves
from operators.relocate import generate_relocation_moves
from operators.swap import generate_swap_moves
from operators.two_opt import generate_two_opt_inter_moves, generate_two_opt_intra_moves
from utils.checker import clone_routes, evaluate_solution, maybe_repair_to_vehicle_limit


def _clean_routes(problem: ProblemInstance, routes: List[List[int]]) -> List[List[int]]:
    return [route for route in routes if len(route) > 2 and route[0] == problem.depot_id and route[-1] == problem.depot_id]


def _total_distance(problem: ProblemInstance, routes: List[List[int]]) -> float:
    feasible, total, _ = evaluate_solution(problem, routes)
    if not feasible:
        return float("inf")
    return total


def _operator_generators():
    return {
        "relocate": generate_relocation_moves,
        "swap": generate_swap_moves,
        "two_opt_intra": generate_two_opt_intra_moves,
        "two_opt_inter": generate_two_opt_inter_moves,
        "or_opt": generate_or_opt_moves,
        "cross_exchange": generate_cross_exchange_moves,
    }


def _generate_candidates(
    problem: ProblemInstance,
    routes: List[List[int]],
    per_operator: int,
    enabled_operators: List[str],
) -> List[MoveCandidate]:
    candidates: List[MoveCandidate] = []
    generators = _operator_generators()

    for operator in enabled_operators:
        if operator not in generators:
            continue
        candidates.extend(generators[operator](problem, routes, max_moves=per_operator))

    for cand in candidates:
        cand.routes = _clean_routes(problem, cand.routes)
        cand.objective = _total_distance(problem, cand.routes)

    candidates.sort(key=lambda c: c.objective)
    return [c for c in candidates if c.objective < float("inf")]


def tabu_search(
    problem: ProblemInstance,
    initial_routes: List[List[int]],
    iterations: int = 150,
    tabu_tenure: int = 12,
    aspiration: bool = True,
    diversification_interval: int = 25,
    intensification_interval: int = 20,
    per_operator_moves: int = 40,
    enabled_operators: Optional[List[str]] = None,
    iteration_callback: Optional[Callable[[Dict], None]] = None,
    apply_fleet_repair: bool = True,
    random_seed: int = 0,
) -> Tuple[List[List[int]], float]:
    rng = random.Random(random_seed)

    current, _ = maybe_repair_to_vehicle_limit(
        problem,
        _clean_routes(problem, clone_routes(initial_routes)),
        apply_fleet_repair=apply_fleet_repair,
    )
    current_cost = _total_distance(problem, current)
    best = clone_routes(current)
    best_cost = current_cost

    tabu: Dict[Tuple, int] = {}
    no_improve = 0
    operators = enabled_operators or [
        "relocate",
        "swap",
        "two_opt_intra",
        "two_opt_inter",
        "or_opt",
        "cross_exchange",
    ]

    if iteration_callback:
        iteration_callback(
            {
                "iteration": 0,
                "event": "start",
                "current_cost": current_cost,
                "best_cost": best_cost,
                "operators": operators,
            }
        )

    for it in range(1, iterations + 1):
        candidates = _generate_candidates(
            problem,
            current,
            per_operator=per_operator_moves,
            enabled_operators=operators,
        )

        chosen: Optional[MoveCandidate] = None
        for cand in candidates:
            is_tabu = tabu.get(cand.move_key, -1) >= it
            can_aspire = aspiration and cand.objective < best_cost
            if not is_tabu or can_aspire:
                chosen = cand
                break

        if chosen is None:
            # Diversification by restart from a random feasible constructor.
            methods = get_initial_methods(seed=rng.randint(0, 10_000))
            restart_routes, _ = maybe_repair_to_vehicle_limit(
                problem,
                _clean_routes(problem, methods["random"](problem)),
                apply_fleet_repair=apply_fleet_repair,
            )
            restart_cost = _total_distance(problem, restart_routes)
            current = restart_routes
            current_cost = restart_cost
            no_improve += 1
            if iteration_callback:
                iteration_callback(
                    {
                        "iteration": it,
                        "event": "restart_no_candidate",
                        "current_cost": current_cost,
                        "best_cost": best_cost,
                        "candidate_count": len(candidates),
                    }
                )
            continue

        current = chosen.routes
        current_cost = chosen.objective
        tabu[chosen.move_key] = it + tabu_tenure
        event = "move"

        if current_cost < best_cost:
            best = clone_routes(current)
            best_cost = current_cost
            no_improve = 0
            event = "improvement"
        else:
            no_improve += 1

        if diversification_interval > 0 and no_improve >= diversification_interval:
            methods = get_initial_methods(seed=rng.randint(0, 10_000))
            current, _ = maybe_repair_to_vehicle_limit(
                problem,
                _clean_routes(problem, methods["random"](problem)),
                apply_fleet_repair=apply_fleet_repair,
            )
            current_cost = _total_distance(problem, current)
            no_improve = 0
            event = "diversification"

        if intensification_interval > 0 and it % intensification_interval == 0:
            current = clone_routes(best)
            current_cost = best_cost
            event = "intensification"

        if iteration_callback:
            iteration_callback(
                {
                    "iteration": it,
                    "event": event,
                    "move_key": list(chosen.move_key),
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "candidate_count": len(candidates),
                    "tabu_size": len(tabu),
                }
            )

    return best, best_cost


def print_solution(method_name: str, routes: List[List[int]], distance: float, feasible: bool, message: str):
    print("=" * 72)
    print(f"Method: {method_name}")
    print(f"Feasible: {feasible}")
    print(f"Total distance: {distance:.2f}" if feasible else "Total distance: inf")
    print(f"Routes: {len(routes)}")
    print(f"Status: {message}")

    for index, route in enumerate(routes, start=1):
        print(f"Route {index}: {' -> '.join(str(node) for node in route)}")


def _print_tabu_iteration(payload: Dict):
    iteration = payload.get("iteration", -1)
    phase = payload.get("event", "unknown")
    current_cost = payload.get("current_cost", float("inf"))
    best_cost = payload.get("best_cost", float("inf"))
    move_key = payload.get("move_key")
    candidate_count = payload.get("candidate_count")

    if isinstance(move_key, list) and move_key:
        move_label = str(move_key[0])
    else:
        move_label = "-"

    extra = []
    if candidate_count is not None:
        extra.append(f"candidates={candidate_count}")
    if "tabu_size" in payload:
        extra.append(f"tabu={payload['tabu_size']}")

    extra_text = f" | {' | '.join(extra)}" if extra else ""
    print(
        f"Iter {iteration:>3} | phase={phase:<15} | move={move_label:<15} | current={current_cost:.2f} | best={best_cost:.2f}{extra_text}"
    )


def _get_problem(instance: str) -> ProblemInstance:
    from utils.parser import parse_instance

    return parse_instance(instance)


def run_tabu_from_method(
    instance: str,
    seed: int,
    method: str,
    iterations: int,
    tabu_tenure: int,
    aspiration: bool,
    diversification_interval: int,
    intensification_interval: int,
    per_operator_moves: int,
    enabled_operators: Optional[List[str]],
    apply_fleet_repair: bool = True,
) -> None:
    problem = _get_problem(instance)
    methods = get_initial_methods(seed=seed)

    if method not in methods:
        raise ValueError(f"Unknown tabu start method: {method}")

    routes = methods[method](problem)
    routes, repair_note = maybe_repair_to_vehicle_limit(problem, routes, apply_fleet_repair=apply_fleet_repair)

    feasible, distance, message = evaluate_solution(problem, routes)
    if repair_note:
        message = f"{message}{repair_note}"
    if not feasible:
        print_solution(f"Tabu start ({method})", routes, distance, feasible, message)
        return

    print(f"Starting tabu from {method} | init_distance={distance:.2f} | routes={len(routes)}")

    best_routes, best_cost = tabu_search(
        problem=problem,
        initial_routes=routes,
        iterations=iterations,
        tabu_tenure=tabu_tenure,
        aspiration=aspiration,
        diversification_interval=diversification_interval,
        intensification_interval=intensification_interval,
        per_operator_moves=per_operator_moves,
        enabled_operators=enabled_operators,
        iteration_callback=_print_tabu_iteration,
        apply_fleet_repair=apply_fleet_repair,
        random_seed=seed,
    )

    feasible, distance, message = evaluate_solution(problem, best_routes)
    if best_cost < distance:
        distance = best_cost

    print_solution(f"Tabu Search ({method})", best_routes, distance, feasible, message)
