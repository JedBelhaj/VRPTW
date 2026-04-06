from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MoveCandidate:
    routes: List[List[int]]
    move_key: Tuple
    objective: float
