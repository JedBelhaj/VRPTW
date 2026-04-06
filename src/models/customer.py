from dataclasses import dataclass


@dataclass(frozen=True)
class Customer:
    id: int
    x: float
    y: float
    demand: int
    ready_time: float
    due_time: float
    service_time: float

    def __repr__(self):
        return f"C{self.id}"