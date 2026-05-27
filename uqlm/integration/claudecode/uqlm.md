---
description: Add uqlm hallucination detection to the current agent code
---

Help the user add uqlm (https://github.com/cvs-health/uqlm) hallucination detection to their agent.

1. Inspect the current file or working directory to detect:
   - Which framework is in use (LangGraph `StateGraph` / `langgraph.graph`, LangChain Runnables / LCEL, AutoGen, CrewAI, or hand-rolled).
   - Where the LLM response is generated (so the uq node can be inserted right after).
   - Whether the LLM exposes token logprobs (informs the scorer choice).

2. Pick a scorer using this decision tree:
   - **Generating code** → `CodeGenUQ`.
   - **Multi-paragraph / multi-claim output** → `LongTextUQ` (default) or `LongTextGraph` / `LongTextQA`.
   - **LLM exposes logprobs (OpenAI, vLLM, local)** → `WhiteBoxUQ`.
   - **Max accuracy, cost OK** → `UQEnsemble`.
   - **Otherwise (default)** → `BlackBoxUQ` with `scorers=["noncontradiction"]`.

3. Wire it in:
   - **LangGraph**: a plain async node function that calls `await scorer.generate_and_score([state["prompt"]], num_responses=5)`, writes `uq_score` into state, and (if conditional re-ask is wanted) `graph.add_conditional_edges` keyed on `state["uq_score"] < threshold`.
   - **LangChain**: a `RunnableLambda` wrapping the same async call, composed into the chain.
   - **Custom**: call `await scorer.generate_and_score(...)` or `scorer.score(responses=..., sampled_responses=...)` directly from the agent loop.

4. Suggest a starting threshold:
   - `noncontradiction` < 0.5 = likely hallucination.
   - `cosine_sim` < 0.7 = flagged.
   - `consistency_and_confidence` < 0.75 = flagged.
   - For production, tune on a labeled hold-out set.

5. Cap any retry/re-ask loop with a counter in state to prevent infinite loops.

6. Verify install with: `python -c "from uqlm import BlackBoxUQ; print('ok')"`.

User context: $ARGUMENTS
