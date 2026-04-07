import math
from typing import Any, Dict, List

from models.problem import ProblemInstance


def _scaled(value: float, scale: int) -> int:
    return int(round(value * scale))


def compare_solution_with_pyvrp(
    problem: ProblemInstance,
    routes: List[List[int]],
    baseline_feasible: bool,
    baseline_distance: float,
    distance_tolerance: float = 1e-2,
    scale: int = 1000,
) -> Dict[str, Any]:
    try:
        import pyvrp
    except ImportError:
        return {
            "pyvrp_status": "not_installed",
            "pyvrp_feasible": None,
            "pyvrp_distance": None,
            "pyvrp_distance_delta": None,
            "pyvrp_matches": False,
            "pyvrp_message": "pyvrp is not installed",
        }

    try:
        model = pyvrp.Model()

        depot_customer = problem.customers[problem.depot_id]
        depot = model.add_depot(
            x=_scaled(depot_customer.x, scale),
            y=_scaled(depot_customer.y, scale),
            tw_early=_scaled(depot_customer.ready_time, scale),
            tw_late=_scaled(depot_customer.due_time, scale),
            name=f"depot_{problem.depot_id}",
        )

        customer_to_visit_idx: Dict[int, int] = {}
        locations_by_id = {problem.depot_id: depot}

        for visit_idx, customer_id in enumerate(sorted(problem.customer_ids), start=1):
            customer = problem.customers[customer_id]
            client = model.add_client(
                x=_scaled(customer.x, scale),
                y=_scaled(customer.y, scale),
                delivery=int(customer.demand),
                service_duration=_scaled(customer.service_time, scale),
                tw_early=_scaled(customer.ready_time, scale),
                tw_late=_scaled(customer.due_time, scale),
                name=f"c_{customer_id}",
            )
            customer_to_visit_idx[customer_id] = visit_idx
            locations_by_id[customer_id] = client

        model.add_vehicle_type(
            num_available=problem.vehicle_count,
            capacity=int(problem.capacity),
            start_depot=depot,
            end_depot=depot,
            unit_distance_cost=1,
            unit_duration_cost=0,
            name="default",
        )

        node_ids = [problem.depot_id] + sorted(problem.customer_ids)
        for from_id in node_ids:
            from_node = locations_by_id[from_id]
            from_customer = problem.customers[from_id]
            for to_id in node_ids:
                to_node = locations_by_id[to_id]
                to_customer = problem.customers[to_id]
                distance = math.hypot(from_customer.x - to_customer.x, from_customer.y - to_customer.y)
                dist_int = _scaled(distance, scale)
                model.add_edge(from_node, to_node, distance=dist_int, duration=dist_int)

        data = model.data()

        pyvrp_routes: List[List[int]] = []
        for route in routes:
            visits: List[int] = []
            for customer_id in route:
                if customer_id == problem.depot_id:
                    continue
                visits.append(customer_to_visit_idx[customer_id])
            if visits:
                pyvrp_routes.append(visits)

        solution = pyvrp.Solution(data, pyvrp_routes)
        pyvrp_feasible = bool(solution.is_feasible())
        pyvrp_distance = float(solution.distance()) / scale

        feasible_match = pyvrp_feasible == baseline_feasible
        if pyvrp_feasible and baseline_feasible:
            distance_delta = abs(pyvrp_distance - baseline_distance)
            distance_match = distance_delta <= distance_tolerance
        else:
            distance_delta = None
            distance_match = True

        matches = feasible_match and distance_match
        status = "match" if matches else "mismatch"
        return {
            "pyvrp_status": status,
            "pyvrp_feasible": pyvrp_feasible,
            "pyvrp_distance": pyvrp_distance,
            "pyvrp_distance_delta": distance_delta,
            "pyvrp_matches": matches,
            "pyvrp_message": "OK" if matches else "Feasibility or distance mismatch",
        }

    except Exception as exc:
        return {
            "pyvrp_status": "error",
            "pyvrp_feasible": None,
            "pyvrp_distance": None,
            "pyvrp_distance_delta": None,
            "pyvrp_matches": False,
            "pyvrp_message": str(exc),
        }