# api/schemas.py

from pydantic import BaseModel

class CompareRequest(BaseModel):
    prompt: str
    model: str = "qwen-1.5b"
    strategies: list[str] = ["greedy", "top_k", "top_p", "beam"]
    max_new_tokens: int = 200

class StrategyResult(BaseModel):
    text: str
    mean_entropy: float
    repetition_rate: float
    num_input_tokens: int
    num_output_tokens: int
    latency_ms: float
    tokens_per_sec: float

class CompareResponse(BaseModel):
    prompt: str
    model: str
    results: dict[str, StrategyResult]