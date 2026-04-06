from dataclasses import dataclass, field
from typing import List


@dataclass
class Route:
    stops: List[int] = field(default_factory=list)
    load: float = 0.0
    distance: float = 0.0