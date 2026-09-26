# jev-router

A from-scratch, numpy-only implementation of the "System One" decision-model
paradigm — non-autoregressive, schema-typed probability heads with
**calibrated confidence and statistically certified act-or-escalate routing**.

This is the shared architecture of **SalesRLAgent** (arXiv
[2503.23303](https://arxiv.org/2503.23303), March 2025 — open model, dataset,
and code) and **Jev** (TypeSafe AI, September 2026). It was published and
open-sourced eighteen months before Jev's announcement; see
**[PRIOR_ART.md](PRIOR_ART.md)** for the timeline and receipts.

## What it does

One forward pass over a conversation's embedding produces a typed decision
with a calibrated probability. A Wilson-certified confidence gate decides:

- **ACT** — confidence ≥ certified τ: automated action, guaranteed accuracy
  target on the auto-acted set (95% confidence lower bound).
- **ESCALATE** — below τ: defer to a human. Never guess.

Everything fails closed: corrupt model, stale thresholds, distribution drift,
NaN inputs, uncertifiable targets — all escalate with an explicit reason.

## Measured results (saas-sales-conversations, n=8,853, company-grouped splits)

| | value |
|---|---|
| conversion accuracy (head) | **76.3%** (LLM baseline on same data: 48–60%) |
| ECE / Brier | 0.067 / 0.166 |
| Wilson-certified 90% gate | 38.8% auto-act @ 92.3% held-out accuracy |
| Wilson-certified 95% gate | 13.4% auto-act @ 97.1% |
| inference latency | 0.004 ms/decision |
| gradient check (finite diff) | 1.2e-10 |

Dataset is synthetic (GPT-4o generated): treat numbers as evidence of the
paradigm, not production proof. Re-approve on real labels before trusting
coverage in production traffic.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install numpy
.venv/bin/python tests.py && .venv/bin/python tests_real.py && .venv/bin/python tests_prod.py

# reproduce the real-data results (downloads ~640MB via windowed sampling):
.venv/bin/python realdata_emb.py 80 emb_sample.npz
.venv/bin/python train_real_emb.py emb_sample.npz
.venv/bin/python approve.py

# production inference (fail-closed):
echo '{"embedding": [3072 floats], "target": 0.90}' | .venv/bin/python infer.py

# ingest real outcome labels and re-certify:
.venv/bin/python ingest.py labels.jsonl real_labels.npz
.venv/bin/python train_real_emb.py real_labels.npz && .venv/bin/python approve.py && .venv/bin/python monitor.py real_labels.npz
```

## Layout

- `jev.py` — schema primitives (Choice / Score / Noul), parallel heads,
  explicit gradients (finite-difference verified)
- `calibration.py` — ECE, Brier, reliability, temperature scaling
- `routing.py` — Wilson-bounded threshold selection + Router
- `train_real_emb.py` — the paper's architecture: frozen 3072-dim embeddings
  → trained head, leakage-safe company-grouped splits
- `approve.py` / `infer.py` / `monitor.py` — production trio: versioned
  fingerprinted thresholds / strict fail-closed entry point / drift monitor
- `ingest.py` — one-command real-label ingestion
- `tests*.py` — 61 tests incl. corrupted-artifact and fail-closed behavior
- `SKILL.md` — full contract and hard-won rules

## Attribution

Paradigm prior art: **SalesRLAgent** (Nandakishor M, 2025) —
[paper](https://arxiv.org/2503.23303) ·
[model](https://huggingface.co/DeepMostInnovations/sales-conversion-model-reinf-learning) ·
[dataset](https://huggingface.co/datasets/DeepMostInnovations/saas-sales-conversations) ·
confidence routing: [arXiv 2510.01237](https://arxiv.org/abs/2510.01237).
Commercial reference: Jev / TypeSafe AI (2026). This repo implements the
shared paradigm from public contracts alone.

## License

MIT. The training dataset is Apache-2.0 via its HF repo.
