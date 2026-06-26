import torch
import torch.nn.functional as F

def token_entropies(scores: tuple) -> list[float]:
    entropies = []
    for logits in scores:
        logits = logits[0]  # removing batch dimensions 
        probs = F.softmax(logits, dim=-1)
        # clamp to avoid log(0)
        probs = torch.clamp(probs, min=1e-10)
        H = -torch.sum(probs * torch.log(probs)).item()
        entropies.append(H)
    return entropies

def mean_entropy(entropies: list[float]) -> float:
    return sum(entropies) / len(entropies) if entropies else 0.0