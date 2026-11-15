import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    strategy_kwargs: dict,
    max_new_tokens: int = 200,
) -> dict:
    """
    Run a single generation and return the generated text,
    raw generation logits, generated token IDs, and runtime metadata.

    The raw logits are intentionally preserved because EntropyLens
    uses them for post-hoc uncertainty analysis.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
            **strategy_kwargs,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    generated_ids = outputs.sequences[0][input_len:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return {
        "text": generated_text,

        # Raw pre-sampling logits for every generated token.
        # These are the foundation for V2 internal signals.
        "scores": outputs.logits,

        # Preserve generated token IDs so future V2 analysis
        # can associate signals with specific generated tokens.
        "generated_ids": generated_ids,

        "latency_ms": (end - start) * 1000,
        "num_input_tokens": input_len,
        "num_output_tokens": len(generated_ids),
    }