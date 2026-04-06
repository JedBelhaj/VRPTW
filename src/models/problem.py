from dataclasses import dataclass
from typing import Dict, List

from models.customer import Customer


@dataclass(frozen=True)
class ProblemInstance:
    name: str
    vehicle_count: int
    capacity: int
    depot_id: int
    customers: Dict[int, Customer]

    @property
    def customer_ids(self) -> List[int]:
        return [cid for cid in self.customers if cid != self.depot_id]
