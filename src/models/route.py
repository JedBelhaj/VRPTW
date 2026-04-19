from dataclasses import dataclass, field


@dataclass
class Route:
    stops: list = field(default_factory=list)
    load: float = 0.0
    distance: float = 0.0
