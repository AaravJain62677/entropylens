# tests/test_entropy.py

import torch
from metrics.entropy import token_entropies, mean_entropy

def test_uniform_distribution_has_max_entropy():
    vocab_size = 100
    uniform_logits = torch.zeros(1, vocab_size)  # softmax → uniform → max entropy
    scores = (uniform_logits,) * 5
    entropies = token_entropies(scores)
    assert len(entropies) == 5
    # all values should be close to log(vocab_size) ≈ 4.60
    for H in entropies:
        assert abs(H - 4.605) < 0.01

def test_deterministic_distribution_has_zero_entropy():
    vocab_size = 100
    peaked_logits = torch.full((1, vocab_size), -1e9)
    peaked_logits[0][42] = 0.0  # one token gets all probability
    scores = (peaked_logits,)
    entropies = token_entropies(scores)
    assert entropies[0] < 0.01

def test_mean_entropy():
    assert mean_entropy([2.0, 4.0, 6.0]) == 4.0
    assert mean_entropy([]) == 0.0