# metrics/entropy.py

import torch
import torch.nn.functional as F

def token_entropies(scores: tuple) -> list[float]:
    """
    scores: tuple of tensors, length = num_generated_tokens
    each tensor shape: (batch=1, vocab_size)
    returns: list of per-token entropy values (nats)
    """
    entropies = []
    for logits in scores:
        logits = logits[0]  # remove batch dim → (vocab_size,)
        probs = F.softmax(logits, dim=-1)
        # clamp to avoid log(0)
        probs = torch.clamp(probs, min=1e-10)
        H = -torch.sum(probs * torch.log(probs)).item()
        entropies.append(H)
    return entropies

def mean_entropy(entropies: list[float]) -> float:
    return sum(entropies) / len(entropies) if entropies else 0.0