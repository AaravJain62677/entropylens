def summarize_latency(latency_ms: float, num_output_tokens: int) -> dict:
    tokens_per_sec = num_output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0
    return {
        "latency_ms": round(latency_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }