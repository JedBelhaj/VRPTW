import random
import time

from heuristics.helpers import clean_routes, generate_candidates, is_inter_route_move, total_distance
from init.methods import get_initial_methods
from operators.improvement import destroy_smallest_route_regret_reinsert
from utils.checker import clone_routes, evaluate_solution
from utils.parser import parse_instance


DEFAULT_OPERATOR_PERCENTAGES = {
    "relocate": 25.0,
    "swap": 20.0,
    "two_opt_intra": 15.0,
    "two_opt_inter": 15.0,
    "or_opt": 15.0,
    "cross_exchange": 10.0,
}


def normalize_move_key(move_key):
    """Builds a stable, less-fragmented tabu key for move memory."""
    if not move_key:
        return ("noop",)

    operator = move_key[0]

    if operator == "relocate" and len(move_key) >= 2:
        # Customer-centric key avoids route-position overfitting.
        return ("relocate", move_key[1])

    if operator == "swap" and len(move_key) >= 3:
        c1, c2 = move_key[1], move_key[2]
        return ("swap", min(c1, c2), max(c1, c2))

    if operator == "two_opt_intra" and len(move_key) >= 2:
        return ("two_opt_intra", move_key[1])

    if operator == "two_opt_inter" and len(move_key) >= 3:
        r1, r2 = move_key[1], move_key[2]
        return ("two_opt_inter", min(r1, r2), max(r1, r2))

    if operator == "or_opt" and len(move_key) >= 2:
        chain = tuple(sorted(move_key[1])) if isinstance(move_key[1], tuple) else (move_key[1],)
        return ("or_opt",) + chain

    if operator == "cross_exchange" and len(move_key) >= 3:
        seg1 = tuple(move_key[1]) if isinstance(move_key[1], tuple) else (move_key[1],)
        seg2 = tuple(move_key[2]) if isinstance(move_key[2], tuple) else (move_key[2],)
        flat = tuple(sorted(seg1 + seg2))
        return ("cross_exchange",) + flat

    return tuple(move_key)


def tabu_search(
    problem,
    initial_routes,
    iterations=150,
    tabu_tenure=12,
    aspiration=True,
    diversification_interval=25,
    intensification_interval=20,
    total_neighbors=300,
    operator_percentages=None,
    enabled_operators=None,
    iteration_callback=None,
    tenure_increase_step=2,
    max_tabu_tenure=None,
    stagnation_top_k=5,
    perturbation_moves=7,
    enable_improvement_operator=True,
    improvement_interval=30,
    regret_k=2,
):
    rng = random.Random()
    search_start = time.perf_counter()

    # --- Initialization ---
    current = clean_routes(problem, clone_routes(initial_routes))
    current_cost = total_distance(problem, current)
    best = clone_routes(current)
    best_cost = current_cost

    tabu = {}  # Stores move_key: iteration_it_expires
    no_improve = 0
    customer_count = len(problem.customer_ids)
    adaptive_tenure_floor = max(20, customer_count // 5)
    base_tabu_tenure = max(1, tabu_tenure, adaptive_tenure_floor)
    active_tabu_tenure = base_tabu_tenure

    total_neighbors = max(1, int(total_neighbors))
    operator_percentages = operator_percentages or DEFAULT_OPERATOR_PERCENTAGES

    adaptive_perturbation_floor = min(10, max(5, customer_count // 10))
    perturbation_moves = max(1, perturbation_moves, adaptive_perturbation_floor)
    
    stagnation_trigger = diversification_interval // 2 if diversification_interval > 0 else 10
    max_tenure = max_tabu_tenure if max_tabu_tenure is not None else base_tabu_tenure * 4

    operators = enabled_operators or [
        "relocate", "swap", "two_opt_intra", "two_opt_inter", "or_opt", "cross_exchange"
    ]

    def emit(payload):
        if iteration_callback:
            iteration_callback(payload)

    # Initial Progress Emit
    emit({
        "iteration": 0, "event": "start", "current_cost": current_cost,
        "best_cost": best_cost, "operators": operators,
        "active_tabu_tenure": active_tabu_tenure, "elapsed_sec": 0.0,
        "current_routes": clone_routes(current), "best_routes": clone_routes(best),
    })

    for it in range(1, iterations + 1):
        periodic_improvement_attempted = False
        periodic_improvement_applied = False
        
        # 1. Periodic Heavy Improvement (Regret Reinsertion)
        if enable_improvement_operator and improvement_interval > 0 and it % improvement_interval == 0:
            periodic_improvement_attempted = True
            improved_routes, improved = destroy_smallest_route_regret_reinsert(
                problem, current, regret_k=regret_k
            )
            if improved:
                improved_cost = total_distance(problem, improved_routes)
                if improved_cost < current_cost:
                    periodic_improvement_applied = True
                    current, current_cost = improved_routes, improved_cost
                    if current_cost < best_cost:
                        best, best_cost = clone_routes(current), current_cost
                        no_improve = 0
                        active_tabu_tenure = base_tabu_tenure
            
            # If this was an improvement-only step, we skip the standard move logic
            emit_payload(it, "improvement_check", None, current_cost, best_cost, tabu, active_tabu_tenure, 0, periodic_improvement_attempted, periodic_improvement_applied, search_start, current, best, "improvement_only", emit)
            continue

        # 2. Neighborhood Search
        candidates = generate_candidates(
            problem,
            current,
            total_moves=total_neighbors,
            enabled_operators=operators,
            operator_percentages=operator_percentages,
        )
        
        # Ensure candidates are sorted by objective (Best to Worst)
        candidates.sort(key=lambda x: x.objective)

        admissible = []
        for cand in candidates:
            tabu_key = normalize_move_key(cand.move_key)
            is_tabu = tabu.get(tabu_key, -1) >= it
            can_aspire = aspiration and cand.objective < (best_cost - 1e-6) # Small epsilon for float comparison
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
        elif candidates:
            # FALLBACK: If all moves are Tabu, pick the one that expires SOONEST (Least Tabu)
            selection_mode = "least_tabu_fallback"
            chosen = min(candidates, key=lambda c: tabu.get(normalize_move_key(c.move_key), 0))
        
        # 3. Apply Move
        if chosen:
            current = chosen.routes
            current_cost = chosen.objective
            tabu_key = normalize_move_key(chosen.move_key)
            tabu[tabu_key] = it + active_tabu_tenure
            tabu_until = tabu[tabu_key]
            
            if current_cost < (best_cost - 1e-6):
                best, best_cost = clone_routes(current), current_cost
                no_improve = 0
                active_tabu_tenure = base_tabu_tenure
                event = "improvement"
            else:
                no_improve += 1
                event = "move"
                # Dynamic Tenure: Increase if stagnating
                if no_improve >= stagnation_trigger:
                    active_tabu_tenure = min(max_tenure, active_tabu_tenure + tenure_increase_step)
        else:
            # Extreme case: No neighbors generated at all
            event = "deadlock_restart"
            methods = get_initial_methods()
            current = clean_routes(problem, methods["random"](problem))
            current_cost = total_distance(problem, current)
            no_improve += 1
            tabu_until = None

        # 4. Diversification (Perturbation)
        if diversification_interval > 0 and no_improve >= diversification_interval:
            event = "diversification_perturbation"
            current = apply_perturbation(
                problem,
                current,
                operators,
                operator_percentages,
                perturbation_moves,
                stagnation_top_k,
                total_neighbors,
                rng,
            )
            current_cost = total_distance(problem, current)
            no_improve = 0
            active_tabu_tenure = base_tabu_tenure

        # 5. Intensification (Return to Best)
        if intensification_interval > 0 and it % intensification_interval == 0 and event != "improvement":
            current, current_cost = clone_routes(best), best_cost
            event = "intensification"

        # Cleanup Tabu list every 100 iters to save memory
        if it % 100 == 0:
            tabu = {k: v for k, v in tabu.items() if v >= it}

        emit_payload(it, event, chosen, current_cost, best_cost, tabu, active_tabu_tenure, len(candidates), periodic_improvement_attempted, periodic_improvement_applied, search_start, current, best, selection_mode, emit)

    return best, best_cost


def apply_perturbation(problem, routes, operators, operator_percentages, moves, top_k_val, total_neighbors, rng):
    """Applies a series of random inter-route moves to jump out of local optima."""
    perturbed = clone_routes(routes)
    for _ in range(max(1, moves)):
        cands = generate_candidates(
            problem,
            perturbed,
            total_moves=max(20, total_neighbors // 2),
            enabled_operators=operators,
            operator_percentages=operator_percentages,
        )
        if not cands: break
        # Prefer inter-route moves for diversification
        inter = [c for c in cands if is_inter_route_move(c.move_key)]
        pool = inter if inter else cands
        top = max(1, min(top_k_val, len(pool)))
        perturbed = pool[rng.randrange(top)].routes
    return perturbed


def emit_payload(it, event, chosen, current_cost, best_cost, tabu, tenure, cand_count, impr_att, impr_app, start, current, best, sel_mode, emit_func):
    move_key = list(chosen.move_key) if chosen else []
    emit_func({
        "iteration": it, "event": event, "move_key": move_key,
        "current_cost": current_cost, "best_cost": best_cost,
        "candidate_count": cand_count, "tabu_size": len(tabu),
        "active_tabu_tenure": tenure, "selection_mode": sel_mode,
        "periodic_improvement_attempted": impr_att,
        "periodic_improvement_applied": impr_app,
        "elapsed_sec": time.perf_counter() - start,
        "current_routes": clone_routes(current), "best_routes": clone_routes(best),
    })


def run_tabu_from_method(
    instance,
    method,
    iterations=150,
    tabu_tenure=12,
    aspiration=True,
    diversification_interval=25,
    intensification_interval=20,
    total_neighbors=300,
    operator_percentages=None,
    enabled_operators=None,
    print_iterations=False,
    iteration_callback=None,
    tenure_increase_step=2,
    max_tabu_tenure=None,
    stagnation_top_k=5,
    perturbation_moves=3,
    enable_improvement_operator=True,
    improvement_interval=30,
    regret_k=2,
):
    """Runs an initial method + tabu search and returns a benchmark-style row."""
    methods = get_initial_methods()
    if method not in methods:
        raise ValueError(f"Unknown initial method: {method}")

    problem = parse_instance(instance)

    init_start = time.perf_counter()
    initial_routes = methods[method](problem)
    initial_routes = clean_routes(problem, clone_routes(initial_routes))
    init_elapsed = time.perf_counter() - init_start

    init_feasible, init_distance, init_message = evaluate_solution(problem, initial_routes)

    def _iteration_bridge(payload):
        if print_iterations:
            it = payload.get("iteration", "?")
            event = payload.get("event", "-")
            current_cost = payload.get("current_cost")
            best_cost = payload.get("best_cost")
            print(
                f"[it={it:>4}] event={event:<28} "
                f"current={current_cost:.5f} best={best_cost:.5f}"
            )
        if iteration_callback:
            iteration_callback(payload)

    tabu_start = time.perf_counter()
    best_routes, best_cost = tabu_search(
        problem=problem,
        initial_routes=initial_routes,
        iterations=iterations,
        tabu_tenure=tabu_tenure,
        aspiration=aspiration,
        diversification_interval=diversification_interval,
        intensification_interval=intensification_interval,
        total_neighbors=total_neighbors,
        operator_percentages=operator_percentages,
        enabled_operators=enabled_operators,
        iteration_callback=_iteration_bridge,
        tenure_increase_step=tenure_increase_step,
        max_tabu_tenure=max_tabu_tenure,
        stagnation_top_k=stagnation_top_k,
        perturbation_moves=perturbation_moves,
        enable_improvement_operator=enable_improvement_operator,
        improvement_interval=improvement_interval,
        regret_k=regret_k,
    )
    tabu_elapsed = time.perf_counter() - tabu_start

    tabu_feasible, tabu_distance, tabu_message = evaluate_solution(problem, best_routes)
    final_distance = best_cost if tabu_feasible else float("inf")
    final_message = tabu_message if tabu_feasible else tabu_message

    return {
        "instance": problem.name,
        "method": method,
        "init_feasible": init_feasible,
        "init_distance": init_distance,
        "init_elapsed_seconds": init_elapsed,
        "tabu_feasible": tabu_feasible,
        "tabu_distance": final_distance,
        "total_elapsed_seconds": init_elapsed + tabu_elapsed,
        "routes_used": len(best_routes),
        "vehicles_available": problem.vehicle_count,
        "capacity": problem.capacity,
        "message": final_message,
        "init_message": init_message,
        "best_routes": clone_routes(best_routes),
    }