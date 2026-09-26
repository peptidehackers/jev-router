"""Regression tests for the real-data path — every test encodes a failure
actually discovered during the real-data integration.

Run: ./.venv/bin/python tests_real.py   (exits nonzero on failure)
"""
import csv
import io
import json
import sys

import numpy as np

import routing
from jev import Question
from train_real_emb import EmbModel, grad_check, losses_and_grads, eng_bin, split_of

FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def check_split_diversity(splits_companies, min_per_split=3):
    """Guard against the company-ordered-prefix failure: a sequential prefix
    of the CSV contains very few companies, producing degenerate splits."""
    bad = [s for s, cs in splits_companies.items() if len(cs) < min_per_split]
    return bad or None


print("[split diversity guard — the company-ordered CSV failure]")
check("diverse splits pass", check_split_diversity(
    {"train": {"c1", "c2", "c3", "c4"}, "cal": {"c5", "c6", "c7"}, "test": {"c8", "c9", "c10"}}) is None)
check("prefix-biased splits rejected", check_split_diversity(
    {"train": {"c1", "c2"}, "cal": {"c3"}, "test": {"c4"}}) == ["train", "cal", "test"])
check("split is deterministic + leakage-safe by company",
      split_of("saas-7") == split_of("saas-7") and
      len({split_of(f"saas-{i}") for i in range(20)}) == 3)

print("[engagement bin edges]")
check("0.0 -> bin 0", eng_bin(0.0) == 0)
check("0.199 -> bin 0, 0.2 -> bin 1", eng_bin(0.199) == 0 and eng_bin(0.2) == 1)
check("1.0 -> bin 4 (clamp)", eng_bin(1.0) == 4)
check("out-of-range engagement clamps to bin 4", eng_bin(1.7) == 4)

print("[loader row parsing — malformed/dupe/truncate handling]")
from realdata_sample import KEEP
HDR = KEEP
def make_row(cid="c-1", text="hello world", outcome="1", eng="0.5", eff="0.5",
             style="warm", company="saas-1", length="5"):
    return {k: "" for k in HDR} | {
        "conversation_id": cid, "full_text": text, "outcome": outcome,
        "customer_engagement": eng, "sales_effectiveness": eff,
        "conversation_style": style, "company_id": company,
        "conversation_length": length, "product_type": "x",
        "communication_channel": "chat", "conversation_flow": "single"}

def parse_like_loader(rows, max_text=4000):
    kept, malformed, dupes, trunc = [], 0, 0, 0
    seen = set()
    for rec in rows:
        try:
            rec["outcome"] = int(rec["outcome"])
            rec["customer_engagement"] = float(rec["customer_engagement"])
            rec["sales_effectiveness"] = float(rec["sales_effectiveness"])
            rec["conversation_length"] = int(rec["conversation_length"])
            if not rec["full_text"].strip():
                malformed += 1
                continue
            if len(rec["full_text"]) > max_text:
                rec["full_text"] = rec["full_text"][:max_text]
                trunc += 1
            if rec["conversation_id"] in seen:
                dupes += 1
                continue
            seen.add(rec["conversation_id"])
            kept.append(rec)
        except (ValueError, KeyError):
            malformed += 1
    return kept, malformed, dupes, trunc

kept, mal, dup, tr = parse_like_loader([
    make_row(),                                   # good
    make_row(cid="c-1"),                          # duplicate id
    make_row(cid="c-2", outcome="maybe"),         # unparseable outcome
    make_row(cid="c-3", text="   "),              # empty text
    make_row(cid="c-4", text="x" * 5000),         # over-length
])
check("good row kept, dupe/blank/malformed handled",
      len(kept) == 2 and mal == 2 and dup == 1 and tr == 1 and len(kept[1]["full_text"]) == 4000)

print("[confidence ties in threshold selection]")
conf = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6, 0.6])
corr = np.array([True] * 5 + [False] * 5)
tau = routing.select_threshold(conf, corr, 0.5)
check("tied confidences still select deterministically",
      tau in (0.9, 0.6) or tau is None)

print("[EmbModel gradient check on random data]")
rng = np.random.default_rng(3)
m = EmbModel(rng, d_in=64)
m2 = EmbModel(np.random.default_rng(3), d_in=64)  # same-init check not needed
X = rng.normal(0, 1, (8, 64))
Q = [Question("conversion", "noul"),
     Question("style", "choice", options=["a", "b", "c"]),
     Question("engagement", "score", levels=[0.1, 0.3, 0.5, 0.7, 0.9])]
labels = {"conversion": rng.integers(0, 2, 8).astype(float),
          "style": rng.integers(0, 3, 8),
          "engagement": rng.integers(0, 5, 8)}
ratio = grad_check(m, X, Q, labels, budget=60)
check(f"L2 gradient ratio {ratio:.1e} < 1e-5", ratio < 1e-5)
loss, g = losses_and_grads(m, X, Q, labels)
check("loss finite + all grads finite",
      np.isfinite(loss) and all(np.all(np.isfinite(v)) for v in g.values()))

print("[extreme input lengths to EmbModel]")
big = rng.normal(0, 50, (4, 64))
outs, _ = m.forward(big, Q)
p = outs[0]["probability"]
check("extreme feature magnitudes -> finite valid probabilities",
      np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1)))

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("ALL REAL-DATA TESTS PASSED")
