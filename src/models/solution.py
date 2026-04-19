from dataclasses import dataclass, field


@dataclass
class Solution:
	routes: list = field(default_factory=list)
	total_distance: float = 0.0
	feasible: bool = True


