from heuristics.tabu_search import run_tabu_from_method as tabu_run_tabu_from_method

# Single-instance run settings
INSTANCE = "R201"
TABU_START_METHOD = "solomon"
TABU_ITERATIONS = 100
TABU_TENURE = 20
TABU_ENABLE_DYNAMIC_TENURE = True
TABU_LIST_MAX_SIZE = None
TABU_ASPIRATION = True
TABU_DIVERSIFICATION_INTERVAL = 35
TABU_INTENSIFICATION_INTERVAL = 15
TABU_TOTAL_NEIGHBORS = 480
TABU_OPERATORS = ["relocate", "swap", "two_opt_intra", "two_opt_inter", "or_opt", "cross_exchange"]
TABU_OPERATOR_PERCENTAGES = {
    "relocate": 25.0,
    "swap": 20.0,
    "two_opt_intra": 15.0,
    "two_opt_inter": 15.0,
    "or_opt": 15.0,
    "cross_exchange": 10.0,
}
ENABLE_IMPROVEMENT_OPERATOR = True
IMPROVEMENT_INTERVAL = 30
IMPROVEMENT_REGRET_K = 2
PRINT_ITERATIONS = True


def main():
    print(f"Running Tabu on {INSTANCE} (start={TABU_START_METHOD})")
    row = tabu_run_tabu_from_method(
        instance=INSTANCE,
        method=TABU_START_METHOD,
        iterations=TABU_ITERATIONS,
        tabu_tenure=TABU_TENURE,
        enable_dynamic_tenure=TABU_ENABLE_DYNAMIC_TENURE,
        max_tabu_list_size=TABU_LIST_MAX_SIZE,
        aspiration=TABU_ASPIRATION,
        diversification_interval=TABU_DIVERSIFICATION_INTERVAL,
        intensification_interval=TABU_INTENSIFICATION_INTERVAL,
        total_neighbors=TABU_TOTAL_NEIGHBORS,
        operator_percentages=TABU_OPERATOR_PERCENTAGES,
        enabled_operators=TABU_OPERATORS,
        print_iterations=PRINT_ITERATIONS,
        enable_improvement_operator=ENABLE_IMPROVEMENT_OPERATOR,
        improvement_interval=IMPROVEMENT_INTERVAL,
        regret_k=IMPROVEMENT_REGRET_K,
    )

    print("\n" + "=" * 70)
    print("Final solution")
    print(f"Instance: {row['instance']}")
    print(f"Start method: {row['method']}")
    print(f"Initial distance: {row['init_distance']:.3f}")
    print(f"Best distance: {row['tabu_distance']:.3f}")
    print(f"Feasible: {row['tabu_feasible']}")
    print(f"Routes used: {row['routes_used']} / {row['vehicles_available']}")
    print(f"Elapsed: {row['total_elapsed_seconds']:.3f}s")
    print(f"Status: {row['message']}")
    print("Best routes:")
    for idx, route in enumerate(row["best_routes"], start=1):
        print(f"  Route {idx}: {' -> '.join(str(node) for node in route)}")


if __name__ == "__main__":
    main()
