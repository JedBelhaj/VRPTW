# VRPTW Heuristics and Tabu Search Benchmark

This repository implements and benchmarks constructive heuristics and a tabu search metaheuristic for the Vehicle Routing Problem with Time Windows (VRPTW), using Solomon benchmark instances.

The project is built for experimentation:

- Generate initial feasible solutions with multiple methods.
- Improve them using tabu search with configurable neighborhoods.
- Compare methods on single instances or across the full archive.
- Export CSV logs and visualization figures for analysis notebooks.
- Optionally cross-check computed feasibility and distance with PyVRP.

## 1) Problem Scope

The code targets the classical Solomon-style VRPTW:

- A single depot (node id 0).
- A fixed fleet size and vehicle capacity.
- Customers with coordinates, demand, service time, and time window.
- Objective: minimize total traveled Euclidean distance.

Feasibility constraints enforced in the project:

- Every route starts and ends at the depot.
- Route load must not exceed vehicle capacity.
- Service at each customer must begin within its time window.
- Each customer must be visited exactly once.
- Number of routes must not exceed available vehicles.

## 2) Repository Layout

```text
heuristiques/
  Archive/                     # Solomon benchmark instances (C, R, RC)
  config/
    parameters.json            # Example config file (currently not wired into main.py)
  src/
    main.py                    # Main entrypoint and benchmark orchestration
    heuristics/
      tabu_search.py           # Tabu search engine and wrapper
    init/
      methods.py               # Initializer registry
      greedy_insertion.py
      solomon_i1.py
      clarke_wright.py
      random_feasible.py
      sweep.py
    operators/
      relocate.py
      swap.py
      two_opt.py
      or_opt.py
      cross_exchange.py
      move_types.py
    models/
      customer.py
      problem.py
      route.py
      solution.py
      vehicle.py
    utils/
      parser.py                # Solomon parser and path resolver
      checker.py               # Feasibility + distance evaluation
      pyvrp_validator.py       # Optional parity check with PyVRP
      plot_benchmark_maps.py   # Route map generator from CSV
      plot_benchmark_comparative.py
      config.py                # JSON config loader with defaults
      run_logger.py            # JSONL logger utility
    results/                   # Generated benchmark CSV and figures
  benchmark_methods_3csv_analysis.ipynb
  results_analysis.ipynb
  requirements.txt
```

## 3) Data and Parsing

Input instances are read from Solomon text files in Archive.

Parsing behavior:

- If you pass an explicit path and it exists, that path is used.
- Otherwise, parser looks up Archive/<instance>.txt.
- The depot is required to be customer id 0.

Core parsed fields:

- Fleet: number of vehicles and capacity.
- Customers: x, y, demand, ready time, due time, service time.

## 4) Solution Representation

Routes are represented as integer lists of node ids.

Example:

```python
[0, 17, 35, 9, 0]
```

A full solution is a list of routes:

```python
[
  [0, 17, 35, 9, 0],
  [0, 5, 12, 44, 0],
]
```

Evaluation utilities in utils/checker.py compute:

- Route feasibility and route distance.
- Whole-solution feasibility and total distance.
- Missing/duplicate customer checks.
- Optional fleet repair attempt when route count exceeds fleet size.

## 5) Initial Construction Methods

The initializer registry in init/methods.py exposes:

- greedy
- solomon
- clarke_wright
- random
- sweep

Detailed behavior:

1. Greedy Insertion

- Builds one route at a time.
- Repeatedly appends the nearest feasible unserved customer.

2. Solomon I1-style Insertion

- Seeds routes by earliest due-time customers.
- Uses insertion cost c1 (distance and optional time-shift term).
- Uses c2 criterion to choose best customer to insert next.

3. Clarke-Wright Savings

- Starts from one-customer routes.
- Sorts savings and merges compatible route ends when feasible.

4. Random Feasible

- Shuffles customers by seed.
- Tries to insert each customer into existing routes, else starts singleton route.

5. Sweep

- Sorts customers by angular position around depot.
- Forms capacity-based clusters, then greedily builds feasible routes inside each cluster.

## 6) Tabu Search Engine

Tabu search is implemented in heuristics/tabu_search.py.

### 6.1 Neighborhood operators

The following operators are supported:

- relocate
- swap
- two_opt_intra
- two_opt_inter
- or_opt
- cross_exchange

Each operator generates feasible move candidates up to a configurable limit per iteration (per_operator_moves).

### 6.2 Move handling

Each candidate has:

- routes: resulting solution.
- move_key: hashable move signature for tabu tracking.
- objective: candidate distance estimate (later re-evaluated globally).

Candidates are sorted by objective and filtered to admissible moves:

- Non-tabu, or
- Tabu but aspiration-allowed (improves global best)

### 6.3 Intensification and diversification

The search includes anti-stagnation mechanisms:

- Adaptive tabu tenure increases when no improvement persists.
- Diversification trigger:
  - perturb current solution with random top-k candidate moves, or
  - restart from a random feasible constructor if perturbation fails.
- Intensification trigger:
  - periodically reset current solution to the best-so-far.

### 6.4 Repair support

Before optimization, initial routes can be passed through fleet repair:

- If route count exceeds fleet size, try reinserting customers from one route into others.
- If successful, route count is reduced.

### 6.5 Periodic improvement operator: destroy smallest route + regret insertion

In addition to neighborhood moves, tabu_search.py now applies a periodic improvement operator every 30 iterations by default.

Logic:

- Select the route with the fewest customers.
- Remove that route and collect its customers.
- Reinsert customers using regret insertion (k=2 by default):
  - For each pending customer, compute best and second-best feasible insertion delta cost.
  - Regret = second_best - best.
  - Insert the customer with maximum regret first.
- Accept the rebuilt solution only if it improves current_cost.

Why this helps:

- It is less myopic than simple greedy reinsertion.
- It prioritizes difficult customers early, which is effective for VRPTW feasibility and quality.

### 6.6 End-to-end algorithm logic

The runtime logic can be read as a two-stage pipeline:

1. Build a feasible initial solution.
2. Improve it with tabu search while preserving feasibility.

Detailed flow:

1. Parse instance and build ProblemInstance

- Read Solomon file.
- Load vehicle_count, capacity, and all customers.

2. Generate initial routes

- Choose one constructor from init/methods.py.
- Output a list of depot-anchored routes.

3. Optional fleet repair

- If routes used exceed available vehicles, attempt route-count reduction by customer reinsertion.

4. Validate baseline

- Check each route feasibility (capacity + time windows).
- Check global constraints (all customers covered exactly once, fleet limit).

5. Start tabu search loop

- Set current = initial, best = initial.
- Keep a tabu dictionary move_key -> expiration_iteration.

6. At each iteration

- Generate candidate moves from enabled operators.
- Keep only feasible candidates.
- Sort by objective cost.
- Apply tabu filter with aspiration override.
- Select a move:
  - normally the best admissible move,
  - or random among top-k under stagnation.
- Update current solution and tabu list.
- Update best if improved.

7. Anti-stagnation control

- If no improvement persists, increase active tabu tenure.
- If diversification threshold is reached:
  - perturb by applying random inter-route/top-k moves, or
  - restart from random feasible constructor.
- Periodically intensify by resetting current to best.

8. Finalize

- Return best_routes and best_cost.
- Re-evaluate with checker for final feasibility and consistent distance.

### 6.7 Why this design works

The method balances exploration and exploitation:

- Exploitation comes from selecting low-cost admissible moves.
- Exploration comes from tabu restrictions, random top-k selection during stagnation, and diversification perturbations/restarts.
- Robustness comes from strict feasibility checks in every operator and in final evaluation.

In short:

- Constructors give a valid starting point quickly.
- Local operators provide improvement power.
- Tabu memory prevents short cycles.
- Diversification avoids long stagnation plateaus.

### 6.8 Pseudocode-level view

Initialization phase:

```text
problem <- parse_instance(instance)
routes <- initial_method(problem)
routes <- maybe_repair_to_vehicle_limit(problem, routes)
check(routes)
```

Tabu phase:

```text
current <- routes
best <- current
tabu <- {}

for it in 1..iterations:
  candidates <- generate_candidates(current, operators)
  candidates <- feasible_only(candidates)
  admissible <- [c not tabu] U [tabu(c) and aspiration_improves_best(c)]

  if admissible empty:
    current <- random_restart_or_diversify()
    continue

  move <- select_best_or_top_k_random(admissible, stagnation_state)
  current <- apply(move)
  tabu[move.key] <- it + active_tabu_tenure

  if cost(current) < cost(best):
    best <- current
    reset_stagnation_and_tenure()
  else:
    update_stagnation_and_tenure()

  if diversification_triggered:
    current <- perturb_or_restart(current)

  if intensification_iteration:
    current <- best

return best
```

## 7) Main Execution Modes

main.py uses hardcoded flags at the top of the file to control behavior.

Primary toggles:

- RUN_DATASET_METHOD_AVERAGE
- RUN_INIT_BENCHMARK
- RUN_TABU_BENCHMARK
- RUN_TABU
- INIT_METHOD
- INSTANCE
- TABU\_\* parameters

### Mode A: Initial method run

- Set INIT_METHOD to one method name, all, or skip.
- Evaluates and prints routes and distance.

### Mode B: Benchmark initial methods only

- Set RUN_INIT_BENCHMARK = True.
- Produces CSV rows with stage = init.

### Mode C: Benchmark tabu from each initializer

- Set RUN_TABU_BENCHMARK = True.
- Runs tabu search once per initializer.
- Produces CSV rows with stage = tabu.

### Mode D: Dataset-level method averages

- Set RUN_DATASET_METHOD_AVERAGE = True.
- Iterates all R/C/RC files in Archive.
- Writes two files:
  - \*\_detailed.csv
  - \*\_summary.csv

## 8) Output Files

Typical outputs in src/results:

- benchmark*<INSTANCE>*<timestamp>.csv
- benchmark\_<...>\_comparative.png
- benchmark\_<...>\_maps/<method>.png
- method*average*<timestamp>\_detailed.csv
- method*average*<timestamp>\_summary.csv

CSV rows may include fields such as:

- instance, seed, benchmark_stage, method
- feasible or tabu_feasible
- distance, init_distance, tabu_distance
- elapsed_seconds, total_elapsed_seconds
- routes_used, vehicles_available, capacity
- init_solution, best_solution
- pyvrp_status and related validator fields

## 9) Visual Analysis Tools

Two plotting utilities can be run directly:

1. Comparative performance figure

```bash
python src/utils/plot_benchmark_comparative.py --csv src/results/benchmark_C201_20260407_174323.csv
```

2. Route maps (init vs best)

```bash
python src/utils/plot_benchmark_maps.py --csv src/results/benchmark_C201_20260407_174323.csv
```

Notebook files are provided for broader analysis:

- benchmark_methods_3csv_analysis.ipynb
- results_analysis.ipynb

## 10) Optional PyVRP Validation

Set in main.py:

- USE_PYVRP_VALIDATOR = True

Behavior:

- Builds a PyVRP model programmatically from current instance and routes.
- Compares feasibility and distance against local checker output.
- Stores match status and deltas in benchmark rows.

If PyVRP is not installed, status is reported as not_installed.

## 11) Setup and Run

### 11.1 Environment setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional dependencies for plotting (if not already installed):

```powershell
pip install matplotlib numpy
```

### 11.2 Run main workflow

Because imports are structured around the src package root, run from src:

```powershell
Set-Location src
..\.venv\Scripts\python.exe main.py
```

Before running, edit flags in src/main.py to select instance, benchmarks, and tabu settings.

## 12) Parameter Reference (main.py)

Core run settings:

- INSTANCE: Solomon instance name (example: R108)
- SEED: random seed
- INIT_METHOD: skip, all, or one of methods

Tabu parameters:

- TABU_ITERATIONS
- TABU_TENURE
- TABU_ASPIRATION
- TABU_DIVERSIFICATION_INTERVAL
- TABU_INTENSIFICATION_INTERVAL
- TABU_PER_OPERATOR_MOVES
- TABU_OPERATORS
- TABU_EXTRA_VERBOSE
- ENABLE_IMPROVEMENT_OPERATOR
- IMPROVEMENT_INTERVAL (for example 30 means iterations 30, 60, 90, ...)
- IMPROVEMENT_REGRET_K

Validation and repair:

- APPLY_FLEET_REPAIR
- USE_PYVRP_VALIDATOR
- PYVRP_DISTANCE_TOLERANCE

Benchmark controls:

- RUN_INIT_BENCHMARK
- RUN_TABU_BENCHMARK
- TABU_BENCHMARK_PRINT_ITERATIONS
- RUN_DATASET_METHOD_AVERAGE

## 13) Notes on config/parameters.json

The repository includes a robust config loader in src/utils/config.py and an example JSON file in config/parameters.json.

Current status:

- main.py presently uses hardcoded constants and does not consume config/parameters.json.
- You can integrate load_config() for command-line or file-driven runs in a next iteration.

## 14) Extending the Project

Recommended extension points:

1. Add a new initializer

- Implement in src/init.
- Register it in init/methods.py.

2. Add a new tabu operator

- Implement generator in src/operators.
- Add to \_operator_generators() in heuristics/tabu_search.py.

3. Add new benchmark metrics

- Extend row dictionaries in main.py and post-processing notebooks.

4. Add CLI arguments

- Wrap current hardcoded flags with argparse and optional JSON config loading.

## 15) Troubleshooting

Import errors for modules like utils or init:

- Run from src directory, or ensure src is on Python path.

Infeasible route or missing customers:

- Verify instance formatting and depot id 0.
- Check if fleet repair is disabled when required.

No benchmark rows produced:

- Ensure at least one benchmark flag is enabled.

PyVRP mismatch:

- Compare distance scaling and tolerance.
- Ensure same travel metric assumptions (Euclidean distance).

## 16) License and Academic Use

This repository appears to be intended for educational/research benchmarking on Solomon VRPTW instances. Add a formal LICENSE file if you plan to distribute or reuse the code externally.
