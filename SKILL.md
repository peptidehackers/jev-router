---
name: jev-router
description: Calibrated schema-typed decision heads with selective routing (Jev/"System One" reference implementation). Use when adding fast non-LLM classification/routing/extraction to a system, when gating LLM calls by confidence to cut cost and hallucination risk, when evaluating models by calibration (ECE/reliability/Brier) instead of accuracy, or when implementing confidence-based act-or-escalate routing with statistically certified thresholds. Triggers - "confidence routing", "calibrated classifier", "route to LLM or act", "ECE", "reliability curve", "selective prediction", "schema-typed output", "Jev", "RLCD".
---

# Jev-Router: calibrated decision heads + selective routing

A zero-dependency (numpy-only) reference implementation of the architecture
sold as Jev / "System One" — the same family as SalesRLAgent
(arXiv 2503.23303): encoder → schema-typed probability heads in ONE
non-autoregressive forward pass, trained with proper scoring rules, gated by
calibrated confidence.

**Files** (resolve relative paths against this skill directory):

- `jev.py` — model: Choice / Score / Noul primitives, one parallel pass,
  explicit gradients (finite-difference verified to 1e-10).
- `calibration.py` — ECE, Brier, reliability curves, temperature scaling.
- `routing.py` — Wilson-bounded threshold selection + Router (act/escalate).
- `data.py` — synthetic ambiguous-ticket dataset (swap for real labels).
- `demo.py`, `routing_demo.py` — end-to-end: train → calibrate → gate.
- `realdata_sample.py` / `realdata_emb.py` — windowed range-samplers for the
  7.17GB saas-sales-conversations CSV (companies are stored in contiguous
  blocks: a sequential prefix is company-biased — sample windows across the
  whole file instead). The `_emb` variant keeps the dataset's own 3072-dim
  embeddings.
- `train_real.py` / `train_real_emb.py` — real-data pipelines with
  company-grouped leakage-safe splits (70/15/15), full metric suite, and
  Wilson-certified gates at 85/90/93/95/99%.
- `tests_real.py` — regression tests for every real failure found
  (company-ordered prefix, malformed/duplicate/truncated records, unseen
  labels, bin edges, gradient check on the embedding model, ties).
- `tests.py` — 24 edge-case tests (empty/OOV/unicode inputs, determinism,
  router boundaries, Wilson properties). Run before modifying anything.

**Run**: needs only numpy ≥ 2.x — no numpy? Run
`python3 -m venv .venv && .venv/bin/pip install numpy`, then use
`.venv/bin/python`. `python tests.py && python tests_real.py` must both print
ALL TESTS PASSED; `python demo.py`, `python routing_demo.py`,
`python train_real_emb.py` (after `python realdata_emb.py 40`) reproduce the
numbers above.
## Real-data results (saas-sales-conversations, n=4424, 20 companies,
group-split; conversion = the paper's core decision)

- Text-hash encoder (train_real.py): conversion accuracy at chance despite
  strong confidence separation — bag-of-words is too weak for this signal.
- **Paper's architecture** (frozen 3072-dim embeddings -> trained head,
  train_real_emb.py): TEST conversion **acc 77.1%**, Brier 0.160, ECE 0.017;
  engagement rubric 63.6%; style 16.5% (weak — mostly not recoverable from
  text). Confidence separation: top-decile 95.5% vs bottom 56.8%.
- Wilson-certified auto-act (fit cal, eval test): **85% -> 49.7% coverage @
  88.8%**; **90% -> 23.8% @ 92.9%**; 93/95/99% uncertifiable at n_cal=1107
  -> escalate all (do not weaken the standard).
- OMP three-path bench (50 held-out conversations, default model):
  LLM-only 60.0% ($0.63, 58s); head-only **78.0%** (0.2ms); head+gate@90%
  68.0% (auto-acted 10/10 correct; $0.59 for the 40 escalations, 52s).
  Honest lesson: the gate improves *guarantees*, not accuracy, when the
  escalation target is weaker than the head — route escalations to a human
## Production contract (fail-closed)


Decision flow for every request:

1. `approve.py` fits Wilson-certified thresholds on calibration evidence and
   writes `approved.json` + `baseline.json`, fingerprinted to the exact model
   (sha256), dataset (sha256), and split assignment. A threshold can never
   silently survive a model or data change — `infer.py` verifies all
   fingerprints before acting.
2. `infer.py` is the ONLY inference entry point. Request:
   `{"embedding": [3072 floats], "target": 0.90}` (target optional, default
   0.90). Response: decision (act|escalate), prediction, calibrated
   probability, confidence, tau, threshold version, model sha, and an
   explicit reason. Every unsafe condition escalates: bad/NaN/Inf/wrong-
   length/out-of-range embeddings, unknown or uncertified targets, missing or
   corrupt model/artifacts, fingerprint mismatch. Confidence equal to tau
   acts (>=). Unsupported schemas escalate.
3. `monitor.py <labels.npz>` measures ECE, Brier, reliability, class balance,
   confidence distribution, accuracy, risk-coverage, and certified coverage at
   85/90/93/95/99% against the approval baseline. It NEVER retrains or
   loosens thresholds; on drift it demands fail-closed escalation until
   re-approval. Real outcome labels are ingested via the same npz contract
   (X/outcome/company_id) — no architectural change needed.
4. **Escalation target**: defer to human review / no automated action. Do NOT
   route escalations to a text-only LLM for this decision — measured on a
   company-stratified held-out bench (50 conversations, 4 companies), the
   default-model LLM scored 58.0% vs the head's 80.0%, and certified
   auto-acts went 12/12 correct. Escalating to a weaker decider reduced
   system accuracy (72.0%). The LLM remains the right tool for open-ended
   reasoning, not this finite-output decision.

Certified thresholds in the current `approved.json` (cal n=2214, Wilson 95%):
85% -> tau 0.790; 90% -> 0.918; 93% -> 0.952; 95% -> 0.976; 99% uncertified
-> mandatory escalation. Dataset is synthetic (GPT-4o): treat all numbers as
synthetic evidence, not production proof; re-approve on real outcome labels
before trusting coverage.

## JEV judgment protocol (jev-judge.py / `jev-judge`)

Bounded decisions only. The agent gathers evidence; JEV judges
 Choice/Score/Noul questions **only in Wilson-certified domains**. Every
 consequential judgment is recorded (question, evidence, result, confidence,
 tau, threshold version, action) to `~/.jev/judgments.jsonl`. Out-of-domain
 questions fail closed: "calibrated judgment unavailable — agent/human
 judgment required." Never guess, never treat an uncertified probability as
 evidence. If verified outcomes conflict with JEV, reality wins (monitor.py
 drift => mandatory fail-closed). Currently certified domain:
 `sales-conversion` (noul). Extend the domain table ONLY via approve.py on
 real labels.

## Hard-won rules (each was a real bug — do not relearn them)

1. **Judge by calibration, not accuracy.** Accuracy is the argmax; the
   product is the probability. Report ECE + reliability curve or you are
   measuring the wrong thing.
2. **Uncertainty must come from input ambiguity, not label flips.** Random
   label noise makes confidence uncorrelated with correctness — selective
   routing then has nothing to route (flat risk-coverage curve). Hard cases
   must be genuinely ambiguous inputs.
3. **Select thresholds by Wilson lower bound, not point estimate.** The
   point estimate is optimistic on finite calibration splits. If the bound
   cannot certify the target, `select_threshold` returns None — escalate
   everything until you have more/better calibration data. Size the cal
   split to the target: certifying a 95% gate needs prefixes at ~98%
   observed accuracy; a 93% gate is certifiable from ~95% prefixes.
4. **Report value-vs-claim**: always show held-out accuracy-on-covered next
   to the certified target so optimism is visible.
5. **Type-safety at output, differentiability in training**: clamp Score
   values to rubric bounds in the output dict but compute the loss on the
   raw value, or the gradient check breaks (a clamped loss is flat outside
   bounds).
6. **Verify hand-rolled gradients**: finite-difference check must reach
   ≤1e-7 L2 ratio. Relative-error-per-element blows up on near-zero-gradient
   elements — use the aggregate L2 ratio in `demo.gradient_check`.

## Reference results (synthetic data, 25% ambiguous inputs, 5% flip)

- Gradient check: 1.4e-10. Reliability: conf 0.45→0.55 acc, 0.95→0.95 (ECE 0.025).
- Certified gate (93% target, Wilson 95%): intent 46% auto-act @ 95.9%
  held-out accuracy, fraud 99.9% @ 93.9%; 54% of all-LLM cost; system
  accuracy 95.5% vs head-only 84.6%.
- Latency: ~0.05 ms/sample single, ~1.3 ms for a 128 batch (one pass).

## Integration pattern

```
state → head(state, schema) → {value, probabilities, confidence}
        confidence ≥ τ (Wilson-certified) → act (no LLM call)
        confidence < τ                     → escalate to LLM / human
```

1. Enumerate structured decisions currently prompted to an LLM and
   regex-scraped; define each as a Question (choice/score/noul).
2. Mine ground-truth labels from logs/outcomes.
3. Train with NLL then Brier (`losses_and_grads(mode=...)`).
4. Fit per-question τ on a HELD-OUT split via `routing.select_threshold`.
5. Deploy `Router.decide()` in front of the LLM path; monitor the
   reliability curve and re-fit τ on drift.

Attribution: architecture reference — Jev/TypeSafe AI (2026), SalesRLAgent
(arXiv 2503.23303, Mar 2025). Implementation verified against its own
gradient check and 24 edge-case tests; numbers above are reproducible from
this directory.
