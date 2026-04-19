import json
from copy import deepcopy
from pathlib import Path


DEFAULT_CONFIG = {
    "mode": "single",
    "instance": "R105",
    "seed": 0,
    "initial_methods": ["greedy", "solomon", "clarke_wright", "random", "sweep"],
    "single_run": {
        "init_method": "greedy",
        "use_tabu": True,
    },
    "tabu": {
        "enabled": True,
        "iterations": 120,
        "tabu_tenure": 12,
        "aspiration": True,
        "diversification_interval": 25,
        "intensification_interval": 20,
        "per_operator_moves": 40,
        "operators": [
            "relocate",
            "swap",
            "two_opt_intra",
            "two_opt_inter",
            "or_opt",
            "cross_exchange",
        ],
    },
    "benchmark": {
        "enabled": False,
        "all_operator_combinations": True,
        "max_operator_combo_size": None,
        "run_tabu": True,
        "output_csv": True,
    },
}


def _deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path):
    path = Path(config_path)
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[2]
        path = root / config_path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return deepcopy(DEFAULT_CONFIG)

    user_config = json.loads(raw)
    return _deep_merge(DEFAULT_CONFIG, user_config)

