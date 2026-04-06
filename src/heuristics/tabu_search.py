import random
import time
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
    extra_verbose: bool = False,
) -> Tuple[List[List[int]], float]:
    rng = random.Random(random_seed)
    search_start = time.perf_counter()

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
                "elapsed_sec": 0.0,
            }
        )

    for it in range(1, iterations + 1):
        candidates = _generate_candidates(
            problem,
            current,
            per_operator=per_operator_moves,
            enabled_operators=operators,
        )

        candidate_rows = []
        for cand in candidates:
            is_tabu = tabu.get(cand.move_key, -1) >= it
            can_aspire = aspiration and cand.objective < best_cost
            candidate_rows.append(
                {
                    "move_key": list(cand.move_key),
                    "objective": cand.objective,
                    "is_tabu": is_tabu,
                    "can_aspire": can_aspire,
                }
            )

        chosen: Optional[MoveCandidate] = None
        for cand, row in zip(candidates, candidate_rows):
            if not row["is_tabu"] or row["can_aspire"]:
                chosen = cand
                break

        top_candidate_rows = candidate_rows[:3]

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
                        "elapsed_sec": time.perf_counter() - search_start,
                    }
                )
            continue

        current = chosen.routes
        current_cost = chosen.objective
        tabu[chosen.move_key] = it + tabu_tenure
        tabu_until = tabu[chosen.move_key]
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
                    "move_objective": chosen.objective,
                    "tabu_until": tabu_until,
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "candidate_count": len(candidates),
                    "tabu_size": len(tabu),
                    "elapsed_sec": time.perf_counter() - search_start,
                    "candidate_rows": top_candidate_rows if extra_verbose else None,
                    "tabu_entries": (
                        [
                            {"move_key": list(move_key), "tabu_until": expiry}
                            for move_key, expiry in sorted(tabu.items(), key=lambda item: item[1])
                            if expiry >= it
                        ]
                        if extra_verbose
                        else None
                    ),
                }
            )

        if extra_verbose:
            print(f"Top 3 candidates ({len(candidates)} total):")
            for index, row in enumerate(top_candidate_rows, start=1):
                status = "TABU" if row["is_tabu"] else "OK"
                if row["can_aspire"]:
                    status = f"{status}/ASP"
                chosen_mark = " <= chosen" if chosen is not None and row["move_key"] == list(chosen.move_key) else ""
                print(f"  {index}. {row['move_key']} | cost={row['objective']:.2f} | {status}{chosen_mark}")

            print(f"Chosen move: {list(chosen.move_key)} | cost={chosen.objective:.2f} | tabu_until={tabu_until}")
            print(f"Now tabu: {list(chosen.move_key)} until iteration {tabu_until}")
            print(f"Active tabu list ({len([expiry for expiry in tabu.values() if expiry >= it])}):")
            for move_key, expiry in sorted(tabu.items(), key=lambda item: item[1]):
                if expiry < it:
                    continue
                print(f"  {list(move_key)} -> tabu_until={expiry}")
            try:
                input("Press Enter for next iteration...")
            except EOFError:
                pass

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
    elapsed_sec = payload.get("elapsed_sec")

    if isinstance(move_key, list) and move_key:
        move_label = str(move_key[0])
    else:
        move_label = "-"

    extra = []
    if candidate_count is not None:
        extra.append(f"candidates={candidate_count}")
    if "tabu_size" in payload:
        extra.append(f"tabu={payload['tabu_size']}")
    if elapsed_sec is not None:
        extra.append(f"t={elapsed_sec:.2f}s")

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
    extra_verbose: bool = False,
) -> None:
    overall_start = time.perf_counter()
    problem = _get_problem(instance)
    methods = get_initial_methods(seed=seed)

    if method not in methods:
        raise ValueError(f"Unknown tabu start method: {method}")

    init_start = time.perf_counter()
    routes = methods[method](problem)
    routes, repair_note = maybe_repair_to_vehicle_limit(problem, routes, apply_fleet_repair=apply_fleet_repair)
    init_elapsed = time.perf_counter() - init_start

    feasible, distance, message = evaluate_solution(problem, routes)
    if repair_note:
        message = f"{message}{repair_note}"
    if not feasible:
        print_solution(f"Tabu start ({method})", routes, distance, feasible, message)
        return

    print(
        f"Starting tabu from {method} | init_distance={distance:.2f} | routes={len(routes)} | init_time={init_elapsed:.3f}s"
    )

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
        extra_verbose=extra_verbose,
    )

    feasible, distance, message = evaluate_solution(problem, best_routes)
    if best_cost < distance:
        distance = best_cost

    total_elapsed = time.perf_counter() - overall_start
    message = f"{message} | total_time={total_elapsed:.3f}s"

    print_solution(f"Tabu Search ({method})", best_routes, distance, feasible, message)
