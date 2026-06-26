# strategies/registry.py

STRATEGIES = {
    "greedy": {
        "do_sample": False,
    },
    "top_k": {
        "do_sample": True,
        "top_k": 50,
        "temperature": 0.8,
    },
    "top_p": {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
    },
    "beam": {
        "do_sample": False,
        "num_beams": 4,
        "early_stopping": True,
    },
}

def get_strategy(name: str) -> dict:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(STRATEGIES)}")
    return STRATEGIES[name]

def list_strategies() -> list[str]:
    return list(STRATEGIES.keys())