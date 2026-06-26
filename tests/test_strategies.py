import pytest
from strategies.registry import get_strategy, list_strategies

def test_list_strategies_returns_all_four():
    strategies = list_strategies()
    assert set(strategies) == {"greedy", "top_k", "top_p", "beam"}

def test_get_strategy_greedy():
    s = get_strategy("greedy")
    assert s["do_sample"] == False

def test_get_strategy_top_k():
    s = get_strategy("top_k")
    assert s["do_sample"] == True
    assert "top_k" in s
    assert "temperature" in s

def test_get_strategy_top_p():
    s = get_strategy("top_p")
    assert s["do_sample"] == True
    assert "top_p" in s
    assert "temperature" in s

def test_get_strategy_beam():
    s = get_strategy("beam")
    assert s["do_sample"] == False
    assert s["num_beams"] >= 2

def test_get_strategy_invalid_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("nonexistent_strategy")