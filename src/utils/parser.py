from pathlib import Path
from typing import Dict, List, Tuple

from models.customer import Customer
from models.problem import ProblemInstance


def _resolve_instance_path(instance: str) -> Path:
    candidate = Path(instance)
    if candidate.exists():
        return candidate

    base = Path(__file__).resolve().parents[2]
    archive_candidate = base / "Archive" / f"{instance}.txt"
    if archive_candidate.exists():
        return archive_candidate

    raise FileNotFoundError(f"Instance not found: {instance}")


def parse_instance(instance: str) -> ProblemInstance:
    path = _resolve_instance_path(instance)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    vehicle_count = 0
    capacity = 0
    start_index = 0

    for i, line in enumerate(lines):
        if line.startswith("VEHICLE") and i + 2 < len(lines):
            vehicle_info = lines[i + 2].split()
            vehicle_count = int(vehicle_info[0])
            capacity = int(vehicle_info[1])

        if line.startswith("CUSTOMER"):
            start_index = i + 2
            break

    customers: Dict[int, Customer] = {}
    for line in lines[start_index:]:
        parts = line.split()
        if len(parts) < 7:
            continue

        customer = Customer(
            id=int(parts[0]),
            x=float(parts[1]),
            y=float(parts[2]),
            demand=int(parts[3]),
            ready_time=float(parts[4]),
            due_time=float(parts[5]),
            service_time=float(parts[6]),
        )
        customers[customer.id] = customer

    if 0 not in customers:
        raise ValueError("Depot with id 0 is missing from instance.")

    return ProblemInstance(
        name=path.stem,
        vehicle_count=vehicle_count,
        capacity=capacity,
        depot_id=0,
        customers=customers,
    )


def parse(instance: str) -> Tuple[int, int, dict, List[dict]]:
    problem = parse_instance(instance)
    customers = [
        {
            "id": c.id,
            "x": c.x,
            "y": c.y,
            "demand": c.demand,
            "ready_time": int(c.ready_time),
            "due_time": int(c.due_time),
            "service_time": int(c.service_time),
        }
        for c in problem.customers.values()
    ]
    customers.sort(key=lambda c: c["id"])
    depot = customers[0]
    return problem.vehicle_count, problem.capacity, depot, customers