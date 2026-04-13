import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def _is_inter_route_move(move_key: Tuple) -> bool:
    if not move_key:
        return False

    operator = move_key[0]
    if operator in ("swap", "two_opt_inter", "cross_exchange"):
        return True
    if operator in ("relocate", "or_opt") and len(move_key) >= 4:
        return move_key[2] != move_key[3]
    return False


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
    tenure_increase_step: int = 2,
    max_tabu_tenure: Optional[int] = None,
    stagnation_top_k: int = 5,
    perturbation_moves: int = 3,
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
    base_tabu_tenure = max(1, tabu_tenure)
    active_tabu_tenure = base_tabu_tenure
    stagnation_trigger = max(5, diversification_interval // 2) if diversification_interval > 0 else max(5, iterations // 5)
    max_tenure = max_tabu_tenure if max_tabu_tenure is not None else base_tabu_tenure * 3
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
                "active_tabu_tenure": active_tabu_tenure,
                "elapsed_sec": 0.0,
                "current_routes": clone_routes(current),
                "best_routes": clone_routes(best),
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

        admissible_candidates: List[MoveCandidate] = [
            cand
            for cand, row in zip(candidates, candidate_rows)
            if not row["is_tabu"] or row["can_aspire"]
        ]

        selection_mode = "greedy_best"
        chosen: Optional[MoveCandidate] = None
        if admissible_candidates:
            if no_improve >= stagnation_trigger and len(admissible_candidates) > 1:
                selection_mode = "stagnation_top_k_random"
                top_k = max(1, min(stagnation_top_k, len(admissible_candidates)))
                chosen = admissible_candidates[rng.randrange(top_k)]
            else:
                chosen = admissible_candidates[0]

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
                        "active_tabu_tenure": active_tabu_tenure,
                        "elapsed_sec": time.perf_counter() - search_start,
                        "current_routes": clone_routes(current),
                        "best_routes": clone_routes(best),
                    }
                )
            continue

        current = chosen.routes
        current_cost = chosen.objective
        tabu[chosen.move_key] = it + active_tabu_tenure
        tabu_until = tabu[chosen.move_key]
        event = "move"

        if current_cost < best_cost:
            best = clone_routes(current)
            best_cost = current_cost
            no_improve = 0
            active_tabu_tenure = base_tabu_tenure
            event = "improvement"
        else:
            no_improve += 1
            if no_improve >= stagnation_trigger:
                active_tabu_tenure = min(max_tenure, active_tabu_tenure + max(1, tenure_increase_step))

        if diversification_interval > 0 and no_improve >= diversification_interval:
            perturbed = clone_routes(current)
            perturb_applied = False

            for _ in range(max(1, perturbation_moves)):
                perturb_candidates = _generate_candidates(
                    problem,
                    perturbed,
                    per_operator=max(10, per_operator_moves // 2),
                    enabled_operators=operators,
                )
                if not perturb_candidates:
                    break

                inter_route = [cand for cand in perturb_candidates if _is_inter_route_move(cand.move_key)]
                pool = inter_route if inter_route else perturb_candidates
                top_k = max(1, min(stagnation_top_k, len(pool)))
                picked = pool[rng.randrange(top_k)]
                perturbed = picked.routes
                perturb_applied = True

            if perturb_applied:
                current = _clean_routes(problem, perturbed)
                current_cost = _total_distance(problem, current)
                event = "diversification_perturbation"
            else:
                methods = get_initial_methods(seed=rng.randint(0, 10_000))
                current, _ = maybe_repair_to_vehicle_limit(
                    problem,
                    _clean_routes(problem, methods["random"](problem)),
                    apply_fleet_repair=apply_fleet_repair,
                )
                current_cost = _total_distance(problem, current)
                event = "diversification_restart"

            no_improve = 0
            active_tabu_tenure = base_tabu_tenure

        if intensification_interval > 0 and it % intensification_interval == 0:
            current = clone_routes(best)
            current_cost = best_cost
            active_tabu_tenure = base_tabu_tenure
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
                    "active_tabu_tenure": active_tabu_tenure,
                    "selection_mode": selection_mode,
                    "elapsed_sec": time.perf_counter() - search_start,
                    "current_routes": clone_routes(current),
                    "best_routes": clone_routes(best),
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
    active_tabu_tenure = payload.get("active_tabu_tenure")
    selection_mode = payload.get("selection_mode")

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
    if active_tabu_tenure is not None:
        extra.append(f"tenure={active_tabu_tenure}")
    if selection_mode:
        extra.append(f"sel={selection_mode}")

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
    print_iterations: bool = True,
    iteration_callback: Optional[Callable[[Dict], None]] = None,
    tenure_increase_step: int = 2,
    max_tabu_tenure: Optional[int] = None,
    stagnation_top_k: int = 5,
    perturbation_moves: int = 3,
) -> Dict[str, Any]:
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
    init_distance = distance
    if repair_note:
        message = f"{message}{repair_note}"
    if not feasible:
        print_solution(f"Tabu start ({method})", routes, distance, feasible, message)
        return {
            "instance": problem.name,
            "method": method,
            "init_feasible": feasible,
            "init_distance": distance,
            "init_elapsed_seconds": init_elapsed,
            "tabu_feasible": feasible,
            "tabu_distance": distance,
            "total_elapsed_seconds": time.perf_counter() - overall_start,
            "init_routes": clone_routes(routes),
            "best_routes": clone_routes(routes),
            "routes": clone_routes(routes),
            "routes_used": len(routes),
            "vehicles_available": problem.vehicle_count,
            "capacity": problem.capacity,
            "message": message,
        }

    print(
        f"Starting tabu from {method} | init_distance={distance:.2f} | routes={len(routes)} | init_time={init_elapsed:.3f}s"
    )

    callback: Optional[Callable[[Dict], None]] = None
    if print_iterations or iteration_callback is not None:
        def _dispatch(payload: Dict):
            if print_iterations:
                _print_tabu_iteration(payload)
            if iteration_callback is not None:
                iteration_callback(payload)

        callback = _dispatch

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
        iteration_callback=callback,
        apply_fleet_repair=apply_fleet_repair,
        random_seed=seed,
        extra_verbose=extra_verbose,
        tenure_increase_step=tenure_increase_step,
        max_tabu_tenure=max_tabu_tenure,
        stagnation_top_k=stagnation_top_k,
        perturbation_moves=perturbation_moves,
    )

    feasible, distance, message = evaluate_solution(problem, best_routes)
    if best_cost < distance:
        distance = best_cost

    total_elapsed = time.perf_counter() - overall_start
    message = f"{message} | total_time={total_elapsed:.3f}s"

    print_solution(f"Tabu Search ({method})", best_routes, distance, feasible, message)
    return {
        "instance": problem.name,
        "method": method,
        "init_feasible": True,
        "init_distance": init_distance,
        "init_elapsed_seconds": init_elapsed,
        "tabu_feasible": feasible,
        "tabu_distance": distance,
        "total_elapsed_seconds": total_elapsed,
        "init_routes": clone_routes(routes),
        "best_routes": clone_routes(best_routes),
        "routes": clone_routes(best_routes),
        "routes_used": len(best_routes),
        "vehicles_available": problem.vehicle_count,
        "capacity": problem.capacity,
        "message": message,
    }
