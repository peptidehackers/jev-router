# SalesRLAgent vs. Jev: Prior-Art Comparison

**Claim (disciplined):** the architectural paradigm TypeSafe AI's Jev (Sept 2026)
popularized — non-autoregressive, encoder-backed, schema-typed probability
outputs with calibrated confidence routing — was published, open-sourced, and
demonstrated by SalesRLAgent eighteen months earlier. This is a claim of
**priority of the paradigm**, not of direct influence: there is no evidence
TypeSafe AI knew of the work, and independent invention remains possible. The
receipts below establish what was public, when.

## Timeline

| Date | Event | Receipt |
|---|---|---|
| 2025-03-30 | **SalesRLAgent paper**: RL over sequence embeddings → turn-by-turn conversion probabilities; explicitly non-LLM-generation, 85 ms vs 3450 ms GPT-4o | arXiv 2503.23303 |
| 2025-05-12 | Model + full training dataset open-sourced (Apache-2.0), incl. the 3072-dim embeddings used for training | HF `DeepMostInnovations/sales-conversion-model-reinf-learning`, `saas-sales-conversations` |
| 2025-05 | PyPI package released | per author's public post |
| 2025-09-23 | **Confidence-aware routing paper**: pre-generation confidence score routes to {local generation, RAG, larger model, human review} — the same act-or-escalate guardrail Jev sells | arXiv 2510.01237 |
| 2026-09 | **Jev announced** ("System One"): parallel sampling, RLCD, Choice/Score/Noul schema primitives, calibrated p=0.8 → ~80% correct. No paper, no open weights, no dataset | TypeSafe AI launch coverage (DataCamp, forkast, et al.) |

## Technical mapping (verified against both sources)

| Property | SalesRLAgent (2025-03) | Jev (2026-09) |
|---|---|---|
| Decoding | non-autoregressive; probabilities, no text generation at decision time | same ("cannot write prose"; type-safe outputs only) |
| Representation | frozen 3072-dim sequence embeddings (Azure OpenAI) | encoder-backed state evaluation |
| Training | RL (PPO) over embeddings against ground-truth outcomes | RLCD — "optimizes probability outputs against ground-truth outcomes" |
| Output | turn-by-turn conversion probability 0.0–1.0 | Choice/Score/Noul with full probability spreads |
| Core selling point | fast, calibrated probability vs slow LLM generation | calibrated confidence; thresholds as deterministic guardrails |
| Confidence routing | 2510.01237: 4-pathway confidence router incl. human review | Jev docs: auto-execute >0.90, route below 0.60 |

The defining property both share: **a probability you can threshold**, produced
by a trained head rather than generated text — with the routing guardrail the
second paper made explicit.

## Independent verification (Exhibit A)

A from-scratch reimplementation of the shared paradigm (numpy-only, gradient-
verified to 1.2e-10, 61 tests) reproduces **both sides' claims from public
contracts alone**:

- Trained on the open dataset's own 3072-dim embeddings: conversion accuracy
  **77.1%**, Brier 0.160, **ECE 0.017** — reliability curve shows confidence ≈
  accuracy at every level (the "p=0.8 → 80% correct" property).
- Wilson-certified auto-act gates: 85% target → 46% coverage @ 88%;
  90% → 23% @ 94% — the exact deterministic-guardrail behavior Jev describes.
- Latency 0.004 ms/decision — the speed claim, without autoregression.

The reimplementation is evidence that the paradigm is fully specified by the
2025 publications plus standard calibration machinery; nothing in Jev's 2026
description requires more than what SalesRLAgent shipped first.

## Where the claims differ (do not overclaim)

- Jev is horizontal (general schema primitives); SalesRLAgent is vertical
  (sales conversion). The shared thing is the paradigm, not the product.
- "RLCD" vs "PPO" are different names for outcome-grounded probability
  optimization; the formalization (Choice/Score/Noul) is TypeSafe's own.
- The 96.7% headline in the paper measures accuracy, not calibration; on a
  company-grouped (leakage-safe) split the same architecture measures 77.1% —
  the honest number, and calibration is the stronger claim anyway.

## Recommended actions

1. Publish this comparison (LocalLLaMA/HN) — let the dates argue.
2. arXiv v2 of 2503.23303: add ECE/reliability curves (exhibit A shows how) —
  lead with calibration, the metric Jev markets.
3. Cite both papers in any Jev-adjacent discussion; the Sept 2025 routing
  paper is the direct prior for confidence-threshold guardrails.
