# api/app.py

from fastapi import FastAPI, HTTPException
from api.schemas import CompareRequest, CompareResponse, StrategyResult
from services.state import model_state
from services.comparator import run_comparison
from utils.loader import load_config

app = FastAPI(title="EntropyLens", version="1.0.0")
config = load_config()

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model_state.is_loaded()}

@app.get("/strategies")
def get_strategies():
    from strategies.registry import list_strategies
    return {"strategies": list_strategies()}

@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    if req.model not in config["models"]:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model}'")

    if not model_state.is_loaded() or model_state.active_model_name != req.model:
        model_cfg = config["models"][req.model]
        model_state.load(req.model, model_cfg["path"], config["inference"]["device"])

    results = run_comparison(
        req.prompt,
        req.strategies,
        model_state.model,
        model_state.tokenizer,
        req.max_new_tokens,
    )

    formatted = {}
    for strategy, data in results.items():
        formatted[strategy] = StrategyResult(
            text=data["text"],
            mean_entropy=data["mean_entropy"],
            repetition_rate=data["repetition_rate"],
            num_input_tokens=data["num_input_tokens"],
            num_output_tokens=data["num_output_tokens"],
            latency_ms=data["latency"]["latency_ms"],
            tokens_per_sec=data["latency"]["tokens_per_sec"],
        )

    return CompareResponse(prompt=req.prompt, model=req.model, results=formatted)