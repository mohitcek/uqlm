---
name: uqlm
description: Use when adding hallucination detection, confidence scoring, or uncertainty quantification to an LLM agent or pipeline (LangGraph, LangChain, AutoGen, CrewAI, or any Python agent). Triggers on phrases like "score the response", "detect hallucination", "check confidence", "flag unreliable answers", "uqlm", "BlackBoxUQ", "WhiteBoxUQ", "LLMPanel", "UQEnsemble", "SemanticEntropy", "CodeGenUQ", "LongTextUQ", or when the user wants an agent to gate, retry, or branch on low-confidence outputs.
---

# uqlm — Hallucination Detection for LLM Agents

`uqlm` is a Python library for **LLM hallucination detection via uncertainty quantification**. It exposes "scorers" that return a confidence score in `[0, 1]` for a given prompt/response — low score ≈ likely hallucination. Use it to gate, retry, or flag agent outputs.

```bash
pip install uqlm
```

All scorers are imported directly: `from uqlm import BlackBoxUQ, WhiteBoxUQ, LLMPanel, UQEnsemble, SemanticEntropy, CodeGenUQ, LongTextUQ, LongTextQA, LongTextGraph`.

---

## When to use this skill

Use this skill when the user is:
- Adding hallucination detection, confidence scoring, or self-consistency checking to an agent.
- Wiring quality gates into a LangGraph / LangChain / AutoGen / CrewAI pipeline.
- Asking which scorer fits their use case.
- Tuning a confidence threshold.

Do not use this skill for general agent debugging, prompt engineering, or latency optimization that doesn't involve uqlm.

---

## Scorer cheat sheet (pick one)

| Use case | Scorer | Cost | Needs logprobs |
|---|---|---|---|
| Generic short-form QA, any LLM | `BlackBoxUQ` | medium (N samples) | no |
| Fast, white-box LLM (OpenAI/local with logprobs) | `WhiteBoxUQ` | low | yes |
| LLM-as-judge ensemble | `LLMPanel` | high | no |
| Maximally accurate (combines scorers) | `UQEnsemble` | high | optional |
| Semantic consistency / cluster entropy | `SemanticEntropy` | medium | no |
| Code generation, self-gating | `CodeGenUQ` | medium | optional |
| Long-form / multi-claim text | `LongTextUQ` | high | no |
| Long-form QA (claim → question → answer) | `LongTextQA` | high | no |
| Long-form via centrality on claim graph | `LongTextGraph` | high | no |

Default scorer when unsure: **`BlackBoxUQ`** with `scorers=["noncontradiction"]`.

---

## Quickstart: score a response

```python
import asyncio
from langchain_openai import ChatOpenAI
from uqlm import BlackBoxUQ

llm = ChatOpenAI(model="gpt-4o-mini")
bbuq = BlackBoxUQ(llm=llm, scorers=["noncontradiction", "cosine_sim"])

async def main():
    result = await bbuq.generate_and_score(
        prompts=["What is the capital of France?"],
        num_responses=5,
    )
    df = result.to_df()
    print(df[["response", "noncontradiction", "cosine_sim"]])

asyncio.run(main())
```

Key facts about the result:
- `result` is a `UQResult`; `result.data` is a dict with keys `prompts`, `responses`, plus one key per requested scorer (e.g. `noncontradiction`, `cosine_sim`).
- `result.to_df()` returns a pandas DataFrame, one row per prompt.
- `generate_and_score` is **async** on every scorer — call it with `await` inside `async def`, or wrap with `asyncio.run`.

---

## LangGraph integration (recommended: plain async node)

A uqlm scorer becomes a LangGraph node by wrapping it in a plain async function — no extra wrapper class required. This keeps the agent code readable and decoupled from uqlm internals.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from uqlm import BlackBoxUQ

class AgentState(TypedDict):
    prompt: str
    response: str
    uq_score: float
    low_confidence: bool

llm = ChatOpenAI(model="gpt-4o-mini")
bbuq = BlackBoxUQ(llm=llm, scorers=["noncontradiction"])

async def generate_node(state: AgentState) -> dict:
    msg = await llm.ainvoke(state["prompt"])
    return {"response": msg.content}

async def uq_node(state: AgentState) -> dict:
    result = await bbuq.generate_and_score(
        prompts=[state["prompt"]],
        num_responses=5,
    )
    score = float(result.data["noncontradiction"][0])
    return {
        "response": result.data["responses"][0],
        "uq_score": score,
        "low_confidence": score < 0.5,
    }

graph = StateGraph(AgentState)
graph.add_node("generate", generate_node)
graph.add_node("uq", uq_node)
graph.set_entry_point("generate")
graph.add_edge("generate", "uq")
graph.add_edge("uq", END)
agent = graph.compile()

# Run
import asyncio
asyncio.run(agent.ainvoke({"prompt": "What is the capital of France?"}))
```

### Conditional re-ask on low confidence

```python
def route_on_confidence(state: AgentState) -> str:
    return "regenerate" if state["low_confidence"] else "done"

graph.add_conditional_edges(
    "uq",
    route_on_confidence,
    {"regenerate": "generate", "done": END},
)
```

Note: in production cap the retry loop with a counter in state to avoid infinite re-asks.

### Reusing the agent's own samples (don't re-generate)

If your graph already produced `response` and a few `sampled_responses` for the same prompt, call `BlackBoxUQ.score` (synchronous, no extra LLM calls):

```python
async def uq_node(state):
    result = bbuq.score(
        responses=[state["response"]],
        sampled_responses=[state["sampled_responses"]],  # list of lists
    )
    return {"uq_score": float(result.data["noncontradiction"][0])}
```

`BlackBoxUQ.score` and `SemanticEntropy.score` are **sync**; every other `.score` method (`WhiteBoxUQ.score`, `LLMPanel.score`, `UQEnsemble.score`, `CodeGenUQ.score`, `LongTextUQ.score`, `LongTextQA.score`, `LongTextGraph.score`) is **async** and must be `await`-ed. All `generate_and_score` methods are async.

---

## LangChain integration (RunnableLambda)

Wrap the scorer as a Runnable so it composes into LCEL chains:

```python
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from uqlm import BlackBoxUQ

llm = ChatOpenAI(model="gpt-4o-mini")
bbuq = BlackBoxUQ(llm=llm, scorers=["noncontradiction"])

async def score_step(prompt: str) -> dict:
    result = await bbuq.generate_and_score(prompts=[prompt], num_responses=5)
    return {
        "prompt": prompt,
        "response": result.data["responses"][0],
        "confidence": float(result.data["noncontradiction"][0]),
    }

uq_runnable = RunnableLambda(score_step)

# Use directly
import asyncio
asyncio.run(uq_runnable.ainvoke("What is the capital of France?"))
```

For chains that have already produced a response, swap `score_step` for a version that calls `bbuq.score(responses=..., sampled_responses=...)`.

---

## Custom / generic agents (AutoGen, CrewAI, hand-rolled loops)

uqlm has no framework dependency. Call the scorer from any async function:

```python
from uqlm import BlackBoxUQ

scorer = BlackBoxUQ(llm=llm, scorers=["noncontradiction"])

async def score(prompt: str, response: str, samples: list[str]) -> float:
    result = scorer.score(responses=[response], sampled_responses=[samples])
    return float(result.data["noncontradiction"][0])
```

Gate the agent loop on the returned score.

---

## Code generation (CodeGenUQ)

For agents that **generate code**, use `CodeGenUQ`. It samples N implementations and measures functional equivalence between them (via `equivalence_llm`) plus embedding similarity using a code-aware sentence transformer.

```python
from uqlm import CodeGenUQ

cguq = CodeGenUQ(
    llm=llm,
    scorers=["functional_equivalence_rate", "cosine_sim"],  # defaults
    language="python",
)

result = await cguq.generate_and_score(
    prompts=["Write a Python function that returns the n-th Fibonacci number."],
    num_responses=5,
)
score = float(result.data["functional_equivalence_rate"][0])
```

Wire into a LangGraph code-generation agent the same way as `BlackBoxUQ`: an async node, score < 0.5 → re-ask.

---

## Long-form outputs (multi-paragraph, multi-claim)

For outputs longer than a sentence or two, short-form scorers give noisy averages. Use the long-form family — they decompose the response into atomic claims and score each one.

```python
from uqlm import LongTextUQ

luq = LongTextUQ(
    llm=llm,
    granularity="claim",   # or "response"
    aggregation="mean",
    response_refinement=True,
)
result = await luq.generate_and_score(prompts=prompts, num_responses=5)
# result.data["claims_data"] is a list-of-lists with per-claim scores
```

Variants:
- `LongTextQA` — decomposes claims → generates questions → scores claim-level QA consistency. Higher cost, more accurate.
- `LongTextGraph` — builds a graph of claims and scores via centrality (default `closeness_centrality`).

---

## Picking a scorer (decision tree)

1. Is the agent generating **code**? → `CodeGenUQ`.
2. Is the response **multi-paragraph / multi-claim**? → `LongTextUQ` (cheapest), `LongTextGraph`, or `LongTextQA` (most accurate).
3. Does the LLM provider expose **token logprobs** (OpenAI, vLLM, local)? → `WhiteBoxUQ` (cheap, fast).
4. Need **maximum accuracy** and willing to pay? → `UQEnsemble`.
5. Want **LLM judges**? → `LLMPanel`.
6. Otherwise → `BlackBoxUQ` (universal default).

---

## Thresholds

Confidence cutoffs depend on the sub-scorer. Reasonable starting points:

| Sub-scorer | Default threshold | Rationale |
|---|---|---|
| `noncontradiction` (Black-box NLI) | `< 0.5` ≈ hallucination | NLI probability of non-contradiction across samples |
| `cosine_sim` (Black-box embedding) | `< 0.7` flagged | Embedding similarity across samples |
| `exact_match` | `< 0.5` flagged | Fraction of samples that exact-match |
| `consistency_and_confidence` (White-box) | `< 0.75` flagged | Joint metric |
| `sequence_probability` (White-box) | domain-specific | Tune on a hold-out set |
| `ensemble_scores` (UQEnsemble) | `< 0.5` flagged | Weighted combination |

For production, tune the threshold on a labeled hold-out set rather than guessing. Sweep thresholds over a held-out set of (prompt, response, correctness) tuples and pick the cutoff that maximizes F1 / minimizes false-negatives at your acceptable false-positive rate.

---

## Pitfalls

- **Async-only:** `generate_and_score()` is async on every scorer. Inside Jupyter use `await` directly; in scripts wrap with `asyncio.run(...)`. The sync `.score()` methods are `BlackBoxUQ.score` and `SemanticEntropy.score`; all other `.score()` methods are async.
- **Cost scales with `num_responses`:** Each unit of `num_responses` adds one LLM call per prompt. Start at 3–5.
- **`WhiteBoxUQ` requires logprobs:** Use `ChatOpenAI` with `logprobs=True` or a vLLM/local model. Most Anthropic chat models do not expose token logprobs.
- **`LLMPanel` cost:** N judges × M prompts × J judge calls each. Use cheap judges (e.g. `gpt-4o-mini`).
- **LangChain LLM required:** Scorers accept any `langchain_core.language_models.BaseChatModel` — `ChatOpenAI`, `ChatAnthropic`, `ChatVertexAI`, `ChatLiteLLM`, etc.
- **`response_refinement=True` (long-form) generates a refined response too** — it's NOT just scoring, it's also rewriting. Set `False` if you only want the score.
- **Sub-scorer names matter:** `scorers=["noncontradiction"]` is valid for `BlackBoxUQ` but **not** for `WhiteBoxUQ`. Each scorer's accepted sub-scorers differ — check the constructor docstring if unsure.

---

## Verify install

```bash
python -c "from uqlm import BlackBoxUQ; print('ok')"
```

If this prints `ok`, the user is ready to wire it into their agent.

---

## When this skill should NOT trigger

- The user is asking about model evaluation in general (BLEU, ROUGE, accuracy) — that's not uqlm.
- The user wants to make their agent faster, smaller, or cheaper.
- The user wants observability/logging — that's a different problem.
- The user is asking about RAG retrieval quality — different toolset.
