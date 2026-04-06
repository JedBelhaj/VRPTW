def parse(file_path):
    customers = []
    vehicle_count = 0
    capacity = 0

    with open(f"Archive/{file_path}.txt", 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # --- VEHICLE INFO ---
    for i, line in enumerate(lines):
        if line.startswith("VEHICLE"):
            vehicle_info = lines[i+2].split()
            vehicle_count = int(vehicle_info[0])
            capacity = int(vehicle_info[1])

        # --- CUSTOMER SECTION ---
        if line.startswith("CUSTOMER"):
            start_index = i + 2  # skip header line
            break

    # --- PARSE CUSTOMERS ---
    for line in lines[start_index:]:
        parts = line.split()

        customer = {
            "id": int(parts[0]),
            "x": float(parts[1]),
            "y": float(parts[2]),
            "demand": int(parts[3]),
            "ready_time": int(parts[4]),
            "due_time": int(parts[5]),
            "service_time": int(parts[6]),
        }

        customers.append(customer)

    # depot = first customer
    depot = customers[0]

    return vehicle_count, capacity, depot, customers

vc, cap, depot, customers = parse("R105")

print(vc, cap)
print(depot)
print(len(customers))