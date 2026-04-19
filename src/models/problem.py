from dataclasses import dataclass

from models.customer import Customer


@dataclass(frozen=True)
class ProblemInstance:
    name: str
    vehicle_count: int
    capacity: int
    depot_id: int
    customers: dict

    @property
    def customer_ids(self):
        return [cid for cid in self.customers if cid != self.depot_id]

