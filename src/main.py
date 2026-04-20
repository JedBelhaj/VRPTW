from heuristics.tabu_search import run_tabu_from_method as tabu_run_tabu_from_method

# Single-instance run settings
INSTANCE = "R201"
TABU_START_METHOD = "solomon"
TABU_ITERATIONS = 500
TABU_TENURE = 20
TABU_ASPIRATION = True
TABU_DIVERSIFICATION_INTERVAL = 35
TABU_INTENSIFICATION_INTERVAL = 15
TABU_PER_OPERATOR_MOVES = 80
TABU_OPERATORS = ["relocate", "swap", "two_opt_intra", "two_opt_inter", "or_opt", "cross_exchange"]
ENABLE_IMPROVEMENT_OPERATOR = True
IMPROVEMENT_INTERVAL = 30
IMPROVEMENT_REGRET_K = 2
PRINT_ITERATIONS = True


def main():
    print(f"Running Tabu on {INSTANCE} (start={TABU_START_METHOD})")
    tabu_run_tabu_from_method(
        instance=INSTANCE,
        method=TABU_START_METHOD,
        iterations=TABU_ITERATIONS,
        tabu_tenure=TABU_TENURE,
        aspiration=TABU_ASPIRATION,
        diversification_interval=TABU_DIVERSIFICATION_INTERVAL,
        intensification_interval=TABU_INTENSIFICATION_INTERVAL,
        per_operator_moves=TABU_PER_OPERATOR_MOVES,
        enabled_operators=TABU_OPERATORS,
        print_iterations=PRINT_ITERATIONS,
        enable_improvement_operator=ENABLE_IMPROVEMENT_OPERATOR,
        improvement_interval=IMPROVEMENT_INTERVAL,
        regret_k=IMPROVEMENT_REGRET_K,
    )


if __name__ == "__main__":
    main()
