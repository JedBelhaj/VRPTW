from dataclasses import dataclass


@dataclass
class MoveCandidate:
    routes: list
    move_key: tuple
    objective: float

