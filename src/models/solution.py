from dataclasses import dataclass, field
from typing import List


@dataclass
class Solution:
	routes: List[List[int]] = field(default_factory=list)
	total_distance: float = 0.0
	feasible: bool = True

