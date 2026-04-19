import random
import time

from heuristics.helpers import clean_routes, generate_candidates, is_inter_route_move, total_distance
from init.methods import get_initial_methods
from operators.improvement import destroy_smallest_route_regret_reinsert
from utils.checker import clone_routes, evaluate_solution


def tabu_search(
    problem,
    initial_routes,
    iterations=150,
    tabu_tenure=12,
    aspiration=True,
    diversification_interval=25,
    intensification_interval=20,
    per_operator_moves=40,
    enabled_operators=None,
    iteration_callback=None,
    random_seed=0,
    tenure_increase_step=2,
    max_tabu_tenure=None,
    stagnation_top_k=5,
    perturbation_moves=3,
    enable_improvement_operator=True,
    improvement_interval=30,
    regret_k=2,
):
    rng = random.Random(random_seed)
    search_start = time.perf_counter()

    current = clean_routes(problem, clone_routes(initial_routes))
    current_cost = total_distance(problem, current)
    best = clone_routes(current)
    best_cost = current_cost

    tabu = {}
    no_improve = 0
    base_tabu_tenure = max(1, tabu_tenure)
    active_tabu_tenure = base_tabu_tenure
    if diversification_interval > 0:
        stagnation_trigger = max(5, diversification_interval // 2)
    else:
        stagnation_trigger = max(5, iterations // 5)
    max_tenure = max_tabu_tenure if max_tabu_tenure is not None else base_tabu_tenure * 3

    operators = enabled_operators or [
        "relocate",
        "swap",
        "two_opt_intra",
        "two_opt_inter",
        "or_opt",
        "cross_exchange",
    ]

    def emit(payload):
        if iteration_callback:
            iteration_callback(payload)

    emit(
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
        periodic_improvement_attempted = False
        periodic_improvement_applied = False

        if enable_improvement_operator and improvement_interval > 0 and it % improvement_interval == 0:
            periodic_improvement_attempted = True
            event = "improvement_regret_reinsert_checked"
            selection_mode = "improvement_only"

            improved_routes, improved = destroy_smallest_route_regret_reinsert(
                problem,
                current,
                regret_k=regret_k,
            )
            if improved:
                improved_cost = total_distance(problem, improved_routes)
                if improved_cost < current_cost:
                    periodic_improvement_applied = True
                    current = improved_routes
                    current_cost = improved_cost
                    event = "improvement_regret_reinsert"
                    if current_cost < best_cost:
                        best = clone_routes(current)
                        best_cost = current_cost
                        no_improve = 0
                        active_tabu_tenure = base_tabu_tenure
                else:
                    no_improve += 1
            else:
                no_improve += 1

            emit(
                {
                    "iteration": it,
                    "event": event,
                    "move_key": [],
                    "move_objective": None,
                    "tabu_until": None,
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "candidate_count": 0,
                    "tabu_size": len(tabu),
                    "active_tabu_tenure": active_tabu_tenure,
                    "selection_mode": selection_mode,
                    "periodic_improvement_attempted": periodic_improvement_attempted,
                    "periodic_improvement_applied": periodic_improvement_applied,
                    "elapsed_sec": time.perf_counter() - search_start,
                    "current_routes": clone_routes(current),
                    "best_routes": clone_routes(best),
                }
            )
            continue

        candidates = generate_candidates(
            problem,
            current,
            per_operator=per_operator_moves,
            enabled_operators=operators,
        )

        admissible = []
        for cand in candidates:
            is_tabu = tabu.get(cand.move_key, -1) >= it
            can_aspire = aspiration and cand.objective < best_cost
            if not is_tabu or can_aspire:
                admissible.append(cand)

        selection_mode = "greedy_best"
        chosen = None
        if admissible:
            if no_improve >= stagnation_trigger and len(admissible) > 1:
                selection_mode = "stagnation_top_k_random"
                top_k = max(1, min(stagnation_top_k, len(admissible)))
                chosen = admissible[rng.randrange(top_k)]
            else:
                chosen = admissible[0]

        if chosen is None:
            methods = get_initial_methods(seed=rng.randint(0, 10_000))
            restart_routes = clean_routes(problem, methods["random"](problem))
            current = restart_routes
            current_cost = total_distance(problem, restart_routes)
            no_improve += 1

            emit(
                {
                    "iteration": it,
                    "event": "restart_no_candidate",
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "candidate_count": len(candidates),
                    "active_tabu_tenure": active_tabu_tenure,
                    "periodic_improvement_attempted": periodic_improvement_attempted,
                    "periodic_improvement_applied": periodic_improvement_applied,
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
                step = max(1, tenure_increase_step)
                active_tabu_tenure = min(max_tenure, active_tabu_tenure + step)

        if diversification_interval > 0 and no_improve >= diversification_interval:
            perturbed = clone_routes(current)
            perturb_applied = False

            for _ in range(max(1, perturbation_moves)):
                perturb_candidates = generate_candidates(
                    problem,
                    perturbed,
                    per_operator=max(10, per_operator_moves // 2),
                    enabled_operators=operators,
                )
                if not perturb_candidates:
                    break

                inter_route = [cand for cand in perturb_candidates if is_inter_route_move(cand.move_key)]
                pool = inter_route if inter_route else perturb_candidates
                top_k = max(1, min(stagnation_top_k, len(pool)))
                picked = pool[rng.randrange(top_k)]
                perturbed = picked.routes
                perturb_applied = True

            if perturb_applied:
                current = clean_routes(problem, perturbed)
                current_cost = total_distance(problem, current)
                event = "diversification_perturbation"
            else:
                methods = get_initial_methods(seed=rng.randint(0, 10_000))
                current = clean_routes(problem, methods["random"](problem))
                current_cost = total_distance(problem, current)
                event = "diversification_restart"

            no_improve = 0
            active_tabu_tenure = base_tabu_tenure

        if intensification_interval > 0 and it % intensification_interval == 0:
            current = clone_routes(best)
            current_cost = best_cost
            active_tabu_tenure = base_tabu_tenure
            event = "intensification"

        emit(
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
                "periodic_improvement_attempted": periodic_improvement_attempted,
                "periodic_improvement_applied": periodic_improvement_applied,
                "elapsed_sec": time.perf_counter() - search_start,
                "current_routes": clone_routes(current),
                "best_routes": clone_routes(best),
            }
        )

    return best, best_cost


def print_solution(method_name, routes, distance, feasible, message):
    print("=" * 72)
    print(f"Method: {method_name}")
    print(f"Feasible: {feasible}")
    print(f"Total distance: {distance:.2f}" if feasible else "Total distance: inf")
    print(f"Routes: {len(routes)}")
    print(f"Status: {message}")

    for index, route in enumerate(routes, start=1):
        print(f"Route {index}: {' -> '.join(str(node) for node in route)}")


def _print_tabu_iteration(payload):
    iteration = payload.get("iteration", -1)
    phase = payload.get("event", "unknown")
    current_cost = payload.get("current_cost", float("inf"))
    best_cost = payload.get("best_cost", float("inf"))
    move_key = payload.get("move_key")
    candidate_count = payload.get("candidate_count")
    elapsed_sec = payload.get("elapsed_sec")
    active_tabu_tenure = payload.get("active_tabu_tenure")
    selection_mode = payload.get("selection_mode")
    periodic_improvement_attempted = payload.get("periodic_improvement_attempted", False)
    periodic_improvement_applied = payload.get("periodic_improvement_applied", False)

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
    if periodic_improvement_attempted:
        extra.append(f"impr={'applied' if periodic_improvement_applied else 'checked'}")

    extra_text = f" | {' | '.join(extra)}" if extra else ""
    print(
        f"Iter {iteration:>3} | phase={phase:<15} | move={move_label:<15} | current={current_cost:.2f} | best={best_cost:.2f}{extra_text}"
    )


def _get_problem(instance):
    from utils.parser import parse_instance

    return parse_instance(instance)


def run_tabu_from_method(
    instance,
    seed,
    method,
    iterations,
    tabu_tenure,
    aspiration,
    diversification_interval,
    intensification_interval,
    per_operator_moves,
    enabled_operators,
    print_iterations=True,
    iteration_callback=None,
    tenure_increase_step=2,
    max_tabu_tenure=None,
    stagnation_top_k=5,
    perturbation_moves=3,
    enable_improvement_operator=True,
    improvement_interval=30,
    regret_k=2,
):
    overall_start = time.perf_counter()
    problem = _get_problem(instance)
    methods = get_initial_methods(seed=seed)

    if method not in methods:
        raise ValueError(f"Unknown tabu start method: {method}")

    init_start = time.perf_counter()
    routes = methods[method](problem)
    routes = clean_routes(problem, routes)
    init_elapsed = time.perf_counter() - init_start

    feasible, distance, message = evaluate_solution(problem, routes)
    init_distance = distance
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

    callback = None
    if print_iterations or iteration_callback is not None:
        def _dispatch(payload):
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
        random_seed=seed,
        tenure_increase_step=tenure_increase_step,
        max_tabu_tenure=max_tabu_tenure,
        stagnation_top_k=stagnation_top_k,
        perturbation_moves=perturbation_moves,
        enable_improvement_operator=enable_improvement_operator,
        improvement_interval=improvement_interval,
        regret_k=regret_k,
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

