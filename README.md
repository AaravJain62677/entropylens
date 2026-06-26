# EntropyLens

```
  ███████╗███╗   ██╗████████╗██████╗  ██████╗ ██████╗ ██╗   ██╗██╗     ███████╗███╗   ██╗███████╗
  ██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██╔═══██╗██╔══██╗╚██╗ ██╔╝██║     ██╔════╝████╗  ██║██╔════╝
  █████╗  ██╔██╗ ██║   ██║   ██████╔╝██║   ██║██████╔╝ ╚████╔╝ ██║     █████╗  ██╔██╗ ██║███████╗
  ██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗██║   ██║██╔═══╝   ╚██╔╝  ██║     ██╔══╝  ██║╚██╗██║╚════██║
  ███████╗██║ ╚████║   ██║   ██║  ██║╚██████╔╝██║        ██║   ███████╗███████╗██║ ╚████║███████║
  ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝        ╚═╝   ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝

                    per-token entropy visualizer · decoding strategy comparator
```

A terminal-based platform that runs the same prompt through multiple decoding strategies simultaneously and visualizes Shannon entropy from raw logit distributions — per token, per strategy, side by side.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange) ![License](https://img.shields.io/badge/License-MIT-green) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

---

## Motivation

There is a specific kind of opacity that comes with sampling from a language model. You see the output. You do not see what was discarded to produce it.

When a language model generates a token, it first produces a probability distribution over its entire vocabulary — tens of thousands of candidates, each with an assigned probability. A decoding strategy then collapses that distribution into one token. Greedy picks the mode. Top-p samples from the smallest subset of tokens whose cumulative probability exceeds a threshold. Beam search maintains multiple candidate sequences and selects the globally optimal one.

The distribution itself — the uncertainty the model felt before the decision — is thrown away.

Shannon entropy measures that uncertainty:

```
H(t) = -∑ p(x) log p(x)
```

High entropy at token `t` means the model was genuinely unsure. Many continuations seemed plausible. Low entropy means it committed confidently. EntropyLens makes this visible — per token, across every decoding strategy, simultaneously — so you can see not just what the model said but how uncertain it was when it said it.

This is not a fine-tuned model. It is not a wrapper around another library. It is a structured experimental platform built from first principles: load any HuggingFace causal LM, run it under multiple decoding regimes, collect raw logit distributions, compute entropy, and surface the results in a clean terminal interface with persistent artifact export.

---

## What EntropyLens Is

A fully layered comparison platform. Concretely:

- Run any prompt through greedy, top-k, top-p, and beam search in a single command
- Compute per-token Shannon entropy from raw pre-sampling logit distributions via `output_logits=True`
- Visualize entropy as color-coded terminal bars: green (confident) → yellow → red (uncertain)
- Collect latency, throughput, and repetition rate alongside entropy for each strategy
- Export timestamped JSON artifacts for every run — compare runs, diff strategies across sessions
- Full CLI (`el compare`, `el list`, `el show`, `el diff`) and HTTP API (`/compare`, `/strategies`)
- YAML-driven configuration — swap models, adjust strategies, change inference params without touching code

Supported models out of the box: **Qwen2.5-1.5B-Instruct** and **Phi-3-mini-4k-instruct**. Any HuggingFace `AutoModelForCausalLM` can be added via one line in `config/settings.yaml`.

---

## Relation to TransformerLens

TransformerLens operate on a model's *internals*. They expose attention heads, residual stream activations, and hidden states via hooks and weight captures. They answer: *what circuit produced this behaviour?*

EntropyLens operates one level above that. It observes the *output distribution* — the logit tensor produced at each generation step — and asks: *how uncertain was the model here, and how does the decoding strategy change that?*

No hooks. No model surgery. Works on any HuggingFace model without modification.

The natural research progression:

```
EntropyLens  →  "the model spikes to H=6.38 at token 8 under greedy"
     ↓
TransformerLens  →  "because attention head 4.2 is attending to the wrong token at that position"
```

EntropyLens is the diagnostic layer that surfaces *where* uncertainty concentrates. TransformerLens and NanoLens are the tools to investigate *why*. They are complementary, not competing.

---

## Related Work and Honest Positioning

EntropyLens is not the first tool to connect entropy and LLM generation. Here is what exists and where EntropyLens sits relative to it.

**Entropy-Lens (Akhmedov et al., 2025 — arXiv:2502.16570)**  
The closest name and the closest concept. This is a research paper (not a standalone tool) that uses entropy to analyze *intermediate layer representations* via the logit lens — it tracks how entropy evolves through the residual stream across layers, not across decoding strategies. It builds on TransformerLens and requires hook-level access. EntropyLens by contrast observes the final output logit distribution and requires no internal access — different question, different layer, different scope.

**LM-Polygraph (Fadeeva et al., 2023)**  
A production-grade uncertainty estimation framework with 40+ UQ methods, benchmark infrastructure, and a web demo. It measures uncertainty to detect hallucinations and evaluate model reliability on tasks. It does not compare decoding strategies or visualize per-token entropy interactively. It is a research benchmark; EntropyLens is an interactive comparator. LM-Polygraph is the right tool if you want rigorous hallucination detection. EntropyLens is the right tool if you want to understand *how sampling strategy shapes the distribution* on a specific prompt.

**Decoding Uncertainty (Hashimoto et al., EMNLP 2025 Findings)**  
The paper closest in research question to EntropyLens — it directly studies how decoding strategies affect uncertainty estimation. It runs experiments on Qwen2.5 and Llama across multiple datasets and decoding strategies. It is a paper with a research codebase, not a usable tool. EntropyLens operationalizes the same question as an interactive local platform: any prompt, any HuggingFace model, results in under 2 minutes.

**GLTR (Gehrmann et al., 2019)**  
Visualizes token-level log-probability to detect machine-generated text. Shares the "per-token distribution visualization" idea but is built for text forensics, not decoding comparison. No strategy comparison, no entropy metric, no artifact export.

**AnimatedLLM (2025)**  
An educational visualization tool that shows how autoregressive decoding works with animated token selection. Precomputed traces, browser-based, focused on teaching. Not interactive on arbitrary prompts, no entropy measurement, no strategy comparison.


### Where EntropyLens fits

| Tool | Per-token entropy | Decoding strategy comparison | Any local model | Terminal/CLI | Artifact export |
|------|:-:|:-:|:-:|:-:|:-:|
| Entropy-Lens (paper) | ✓ (layer-level) | ✗ | ✗ | ✗ | ✗ |
| LM-Polygraph | ✓ (sequence-level) | Partial | ✓ | ✗ | ✓ |
| Decoding Uncertainty | ✓ | ✓ | ✓ | ✗ | Research only |
| GLTR | ✓ (log-prob) | ✗ | ✗ | ✗ | ✗ |
| AnimatedLLM | ✗ | ✗ | ✗ | ✗ | ✗ |
| **EntropyLens** | **✓ (token-level)** | **✓** | **✓** | **✓** | **✓** |

The gap EntropyLens fills: an interactive, local, CLI-first platform that runs any prompt through multiple decoding strategies simultaneously and surfaces per-token entropy from raw logits in real time. No server required. No precomputed traces. No paper to read first.

The honest caveat: EntropyLens does not implement the rigorous uncertainty estimation methods that LM-Polygraph does (semantic uncertainty, mutual information, conformal prediction). If your goal is hallucination detection or UQ benchmarking, use LM-Polygraph. If your goal is understanding how greedy differs from top-p on *your* prompt on *your* model, EntropyLens is the faster path.


## Quickstart

```bash
git clone https://github.com/AaravJain62677/entropylens
cd entropylens
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install -e ".[dev]"
```

Run a comparison:

```bash
python -m cli.main compare -p "Once upon a time in a land far away,"
```

---

## Example Output

```
                      Decoding Strategy Comparison
╭──────────┬───────────────┬──────────────┬────────────┬──────────────┬─────────────────╮
│ Strategy │ Output tokens │ Latency (ms) │ Tokens/sec │ Mean entropy │ Repetition rate │
├──────────┼───────────────┼──────────────┼────────────┼──────────────┼─────────────────┤
│ greedy   │           200 │     19060.69 │      10.49 │        1.017 │           0.453 │
│ top_k    │           200 │     17999.05 │      11.11 │        1.414 │           0.257 │
│ top_p    │           200 │     17901.17 │      11.17 │        1.205 │           0.301 │
│ beam     │           200 │     18352.24 │       10.9 │        1.013 │           0.439 │
╰──────────┴───────────────┴──────────────┴────────────┴──────────────┴─────────────────╯

Per-token entropy (■ = high uncertainty)

greedy
  t01 ■■■■■■■■■ 2.32
  t02 ■■■■■ 1.37
  t03 ■ 0.48
  t04 ■■■■■■■■■■■■■■■■■ 4.26   <- genuine branching point, many story directions viable
  t05 ■■■■■■■■■■ 2.68
  t06  0.05                     <- committed, next token nearly certain
  t07 ■■■■ 1.21
  t08 ■■■■■■■■■■■■■■■■■■■■■■■■■ 6.38   <- maximum uncertainty spike

top_k
  t01 ■■■■■■■■■ 2.32
  t02 ■■■■■ 1.37
  t03 ■■■■■■ 1.51               <- diverges from greedy here, sampling explores
  t04 ■■■■■■■■■■■■■■■■■■ 4.58
  t05 ■■■■■■■■■■■■■■■■ 4.03    <- stays in high-entropy region greedy escaped
  ...

Run saved → artifacts/runs/run_2026-06-26_16-07-26.json
```

---

## Research Findings

*Findings are from Qwen2.5-1.5B-Instruct across 5 prompts covering creative, factual, and technical text. Directional and exploratory — not claims about transformers in general.*

### The central finding

**Beam search is confidently repetitive on open-ended generation.**

On the poem prompt ("Write a short poem about the moon"), beam search produced a repetition rate of **0.651** versus top-p's **0.172** — nearly 4× more repetitive — while simultaneously having the *lowest* mean entropy of all four strategies. It was not uncertain. It was wrong in the same way, repeatedly.

This is the classic beam search failure mode on open-ended generation, now measurable with a single command.

### Entropy spikes mark genuine branching points

High entropy in greedy output is not noise. It corresponds to positions where the story, explanation, or argument could legitimately go multiple directions. Token 8 under greedy on the open-ended creative prompt hit **H=6.38** — the model had no confident continuation. Greedy collapsed that uncertainty into one arbitrary choice. Top-k and top-p stayed in that high-entropy region and explored it.

### Prompt type drives mean entropy more than strategy

| Prompt | Greedy H | Top-p H | Beam H |
|--------|----------|---------|--------|
| Poem (creative) | 1.697 | 2.055 | 1.563 |
| Attention mechanisms (technical) | 0.936 | 1.259 | 0.868 |
| Water cycle (constrained) | 0.700 | 0.833 | 0.585 |
| Inflation (factual) | 0.825 | 0.731 | 0.616 |

Creative prompts produce higher mean entropy across *all* strategies than factual ones. The model is more uncertain about how to write a poem than how to explain the water cycle — regardless of whether you sample from it.

### Beam search never truly commits

Beam's entropy bars show large late-sequence spikes (H=6.91, H=7.05 on some prompts) despite being the most deterministic strategy. It forces the model down paths it was not confident about, producing uncertainty spikes in the middle and end of sequences. Greedy commits early and produces smoother entropy curves. Beam optimises globally and pays a per-token uncertainty cost late.

---

## Architecture

EntropyLens follows a strict layered architecture. Business logic lives once, in `services/`. The CLI and API are both thin wrappers over that shared layer.

```
entropylens/
│
├── config/settings.yaml       # models, strategies, inference params — nothing hardcoded
│
├── strategies/registry.py     # strategy name → model.generate() kwargs
│
├── inference/
│   ├── loader.py              # AutoModelForCausalLM + tokenizer loading
│   └── runner.py              # generate() wrapper, returns text + raw logits
│
├── metrics/
│   ├── entropy.py             # per-token Shannon entropy from logit tensors
│   ├── latency.py             # timing + tokens/sec
│   └── quality.py             # BLEU, repetition rate
│
├── services/
│   ├── state.py               # active model singleton (shared by CLI + API)
│   └── comparator.py          # orchestrates N strategies → N result dicts
│
├── display/terminal.py        # rich comparison table + color-coded entropy bars
│
├── tracking/
│   ├── writer.py              # timestamped JSON artifact export
│   └── report.py              # load, list, diff past runs
│
├── cli/main.py                # Typer: compare, list, show, diff
│
├── api/
│   ├── app.py                 # FastAPI: /compare, /strategies, /
│   └── schemas.py             # Pydantic request/response models
│
└── tests/
    ├── test_entropy.py        # entropy math unit tests (3/3 passing)
    ├── test_strategies.py
    └── test_comparator.py
```

The `runner.py` uses `output_logits=True` in `model.generate()` — this returns truly raw pre-sampling logits before any top-k/top-p filtering is applied, ensuring entropy measurements reflect the model's actual internal distribution rather than the post-filtered one.

---

## CLI Reference

```bash
# Compare all 4 strategies on a prompt (default model: qwen-1.5b)
python -m cli.main compare -p "Your prompt here"

# Specify model and subset of strategies
python -m cli.main compare -p "Your prompt" -m phi-3-mini -s greedy -s top_p

# List all saved run artifacts
python -m cli.main list

# Inspect a saved run as JSON
python -m cli.main show run_2026-06-26_16-07-26.json

# Diff two runs side by side
python -m cli.main diff -a run_XXXX.json -b run_YYYY.json
```

---

## HTTP API

```bash
uvicorn api.app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain recursion", "strategies": ["greedy", "top_p"]}'
```

Swagger UI at `http://127.0.0.1:8000/docs`.

---

## Supported Models

| Model | HuggingFace ID | VRAM |
|-------|---------------|------|
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | ~3.5GB |
| Phi-3-mini-4k-instruct | `microsoft/Phi-3-mini-4k-instruct` | ~4.5GB |

Any `AutoModelForCausalLM` model can be added via `config/settings.yaml` — no code changes required.

---

## Supported Decoding Strategies

| Strategy | Sampling | Key parameters |
|----------|----------|---------------|
| Greedy | No | Always picks argmax |
| Top-k | Yes | `top_k=50, temperature=1.2` |
| Top-p (nucleus) | Yes | `top_p=0.9, temperature=1.2` |
| Beam search | No | `num_beams=4, early_stopping=True` |

---

## Limitations (V1)

- One active model at a time — no concurrent inference
- Single GPU only, no multi-GPU support
- Entropy computed from `output_logits=True` — reflects the model's raw distribution but instruction-tuned models have already shaped their logit distributions during RLHF, which compresses the entropy range compared to base models
- No browser UI — terminal and HTTP only
- Artifact storage is local filesystem only
- No activation patching — findings are correlational, not causal

---

## Planned (V2)

- TransformerLens integration: hook into attention heads at high-entropy token positions — the natural next step from "where is the model uncertain" to "why"
- HTML entropy heatmap export, colour-coded per token in browser
- Multi-model parallel comparison: same prompt, same strategy, different models
- Temperature sweep mode: hold strategy fixed, vary temperature, plot entropy curve

---

## Running Tests

```bash
python -m pytest tests/ -v
```

```
tests/test_entropy.py::test_uniform_distribution_has_max_entropy PASSED
tests/test_entropy.py::test_deterministic_distribution_has_zero_entropy PASSED
tests/test_entropy.py::test_mean_entropy PASSED
3 passed in 0.38s
```

---

## Repository Structure

```
entropylens/
├── config/
│   └── settings.yaml
├── strategies/
│   └── registry.py
├── inference/
│   ├── loader.py
│   └── runner.py
├── metrics/
│   ├── entropy.py
│   ├── latency.py
│   └── quality.py
├── services/
│   ├── state.py
│   └── comparator.py
├── display/
│   └── terminal.py
├── tracking/
│   ├── writer.py
│   └── report.py
├── cli/
│   └── main.py
├── api/
│   ├── app.py
│   └── schemas.py
├── tests/
├── data/
│   └── prompts.json
├── artifacts/            # gitignored, generated at runtime
├── utils/
│   └── loader.py
├── pyproject.toml
└── README.md
```

---

## References

- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Elhage et al. — [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- Holtzman et al. — [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (the paper that introduced nucleus/top-p sampling)
- Neel Nanda — [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)

---

## Built By

**Aarav Jain**
[GitHub](https://github.com/AaravJain62677) · [Blog](https://aaravjain.hashnode.dev)

---

## License

MIT — see LICENSE for details.
