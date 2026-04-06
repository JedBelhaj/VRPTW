from typing import List
import time

from heuristics.tabu_search import print_solution as tabu_print_solution
from heuristics.tabu_search import run_tabu_from_method as tabu_run_tabu_from_method
from init.methods import get_initial_methods
from utils.checker import evaluate_solution, maybe_repair_to_vehicle_limit
from utils.parser import parse_instance


ALL_METHODS = ["greedy", "solomon", "clarke_wright", "random", "sweep"]

# Hardcoded settings
INSTANCE = "C105"
SEED = 0
INIT_METHOD = "skip"  # "skip" to skip "all" or one of: greedy, solomon, clarke_wright, random, sweep
APPLY_FLEET_REPAIR = True
RUN_TABU = True
TABU_START_METHOD = "solomon"
TABU_ITERATIONS = 100
TABU_TENURE = 12
TABU_ASPIRATION = True
TABU_DIVERSIFICATION_INTERVAL = 25
TABU_INTENSIFICATION_INTERVAL = 20
TABU_PER_OPERATOR_MOVES = 40
TABU_OPERATORS = ["relocate", "swap", "two_opt_intra", "two_opt_inter", "or_opt", "cross_exchange"]
TABU_EXTRA_VERBOSE = True


def _get_problem(instance: str):
    return parse_instance(instance)


def _get_methods(seed: int):
    return get_initial_methods(seed=seed)


def _run_single_initial(problem, methods, method_name: str):
    start = time.perf_counter()
    routes = methods[method_name](problem)
    routes, repair_note = maybe_repair_to_vehicle_limit(problem, routes, apply_fleet_repair=APPLY_FLEET_REPAIR)
    feasible, distance, message = evaluate_solution(problem, routes)
    elapsed = time.perf_counter() - start
    if repair_note:
        message = f"{message}{repair_note}"
    message = f"{message} | time={elapsed:.3f}s"
    tabu_print_solution(method_name, routes, distance, feasible, message)


def run_initial_solutions(instance: str, seed: int, method: str):
    if method == "skip":
        print("Skipping initial solution generation.")
        return

    problem = _get_problem(instance)
    methods = _get_methods(seed)

    if method == "all":
        selected = [name for name in ALL_METHODS if name in methods]
    else:
        if method not in methods:
            raise ValueError(f"Unknown init method: {method}")
        selected = [method]

    print(f"Instance: {problem.name}")
    print(f"Vehicles available: {problem.vehicle_count}")
    print(f"Capacity: {problem.capacity}")

    for name in selected:
        _run_single_initial(problem, methods, name)


def main():
    run_initial_solutions(instance=INSTANCE, seed=SEED, method=INIT_METHOD)

    if RUN_TABU:
        tabu_run_tabu_from_method(
            instance=INSTANCE,
            seed=SEED,
            method=TABU_START_METHOD,
            iterations=TABU_ITERATIONS,
            tabu_tenure=TABU_TENURE,
            aspiration=TABU_ASPIRATION,
            diversification_interval=TABU_DIVERSIFICATION_INTERVAL,
            intensification_interval=TABU_INTENSIFICATION_INTERVAL,
            per_operator_moves=TABU_PER_OPERATOR_MOVES,
            enabled_operators=TABU_OPERATORS,
            apply_fleet_repair=APPLY_FLEET_REPAIR,
            extra_verbose=TABU_EXTRA_VERBOSE,
        )


if __name__ == "__main__":
    main()
