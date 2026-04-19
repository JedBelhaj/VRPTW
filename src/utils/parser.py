from pathlib import Path

from models.customer import Customer
from models.problem import ProblemInstance


def parse_instance(instance):
    path = Path(instance)
    if not path.exists():
        base = Path(__file__).resolve().parents[2]
        path = base / "Archive" / f"{instance}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Instance not found: {instance}")

    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    vehicle_count = 0
    capacity = 0
    start_index = None

    for i, line in enumerate(lines):
        if line.startswith("VEHICLE") and i + 2 < len(lines):
            parts = lines[i + 2].split()
            vehicle_count = int(parts[0])
            capacity = int(parts[1])
        if line.startswith("CUSTOMER"):
            start_index = i + 2
            break

    if start_index is None:
        raise ValueError("CUSTOMER section not found in instance file.")

    customers = {}
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


def parse(instance):
    problem = parse_instance(instance)
    customers = []
    for c in problem.customers.values():
        customers.append(
            {
                "id": c.id,
                "x": c.x,
                "y": c.y,
                "demand": c.demand,
                "ready_time": int(c.ready_time),
                "due_time": int(c.due_time),
                "service_time": int(c.service_time),
            }
        )
    customers.sort(key=lambda c: c["id"])
    depot = customers[0]
    return problem.vehicle_count, problem.capacity, depot, customers
