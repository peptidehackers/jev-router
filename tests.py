"""Edge-case test suite for the Jev/"System One" reference implementation.

Run: ./.venv/bin/python tests.py   (exits nonzero on any failure)
"""
import sys

import numpy as np

import data
import jev
import routing
from jev import Model, Question

RNG = np.random.default_rng(42)
QUESTIONS = [
    Question("intent", "choice", options=data.INTENTS),
    Question("urgency", "score", levels=[1, 2, 3, 4, 5]),
    Question("is_fraudulent", "noul"),
]
FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


model = Model(np.random.default_rng(7), d_emb=32, d_hidden=48, d_q=24, d_head=32)

print("[adversarial inputs]")
ADVERSARIAL = [
    ("empty string", ""),
    ("whitespace only", "   \n\t  "),
    ("unseen vocab", "zzqxj bwkfn vqrtz plmwb"),
    ("unicode", "会議の件について相談したい 注文の支払い"),
    ("single word", "refund"),
    ("very long", "urgent crash " * 5000),
    ("punctuation only", "!!! ??? ... ,,,"),
]
for label, text in ADVERSARIAL:
    try:
        outs, _, _ = model.forward([text], QUESTIONS)
        by = {o["name"]: o for o in outs}
        ok = (
            np.all(np.isfinite(by["intent"]["probs"]))
            and abs(by["intent"]["probs"].sum() - 1.0) < 1e-6
            and 0 <= int(by["intent"]["choice"][0]) < 4
            and np.all((by["is_fraudulent"]["probability"] >= 0)
                       & (by["is_fraudulent"]["probability"] <= 1))
            and 1.0 <= float(by["urgency"]["value"][0]) <= 5.0
            and abs(by["urgency"]["distribution"].sum() - 1.0) < 1e-6
        )
        check(f"forward survives + valid types: {label}", ok)
    except Exception as e:
        check(f"forward survives + valid types: {label} (raised {e})", False)

print("[determinism]")
o1, _, _ = model.forward(["billing invoice charge card", "crash bug error login"], QUESTIONS)
o2, _, _ = model.forward(["billing invoice charge card", "crash bug error login"], QUESTIONS)
check("identical inputs -> identical probabilities",
      np.array_equal(o1[0]["probs"], o2[0]["probs"])
      and np.array_equal(o1[2]["probability"], o2[2]["probability"]))

print("[batch shapes]")
for n in (1, 2, 500):
    outs, _, _ = model.forward(["invoice charge payment"] * n, QUESTIONS)
    check(f"batch n={n} returns n rows", len(outs[0]["choice"]) == n)

print("[duplicates and mixed content]")
outs, _, _ = model.forward(["refund refund refund refund refund refund", ""], QUESTIONS)
check("duplicated tokens valid",
      abs(outs[0]["probs"].sum(1)[0] - 1.0) < 1e-6 and np.all(np.isfinite(outs[0]["probs"])))

print("[router edges]")
r_none = routing.Router({"intent": None})
check("tau=None escalates everything", not r_none.decide("intent", [0.99, 1.0]).any())
r_zero = routing.Router({"intent": 0.0})
check("tau=0 acts on everything", r_zero.decide("intent", [0.01, 0.0]).all())
check("conf == tau acts (>=)", routing.Router({"i": 0.5}).decide("i", [0.5])[0])
check("empty conf array ok", routing.Router({"i": 0.5}).decide("i", []).shape == (0,))

print("[threshold selection edges]")
conf = np.array([0.9, 0.8, 0.7, 0.6])
corr = np.array([True, True, True, False])
check("impossible target -> None (escalate all)",
      routing.select_threshold(conf, corr, 1.0) is None)
conf_all = np.linspace(0.99, 0.5, 100)   # 100/100 correct, spread confidences
corr_all = np.ones(100, bool)
check("all-correct at n=100 -> tau = lowest confidence",
      abs(routing.select_threshold(conf_all, corr_all, 0.9) - 0.5) < 1e-9)
check("wilson refuses tiny samples (3/3 perfect != 90% certified)",
      routing.select_threshold(np.array([0.9, 0.8, 0.7]),
                               np.array([True, True, True]), 0.9) is None)
# Wilson tau must never be LOWER than the point-estimate tau (conservative)
conf_n = np.concatenate([np.full(60, 0.95), np.full(40, 0.6)])
corr_n = np.concatenate([np.ones(60, bool),
                         np.array([True] * 34 + [False] * 6)])
order = np.argsort(-conf_n, kind="stable")
c, k = conf_n[order], corr_n[order]
accs = np.cumsum(k) / np.arange(1, len(c) + 1)
point_tau = float(c[np.where(accs >= 0.9)[0][-1]])
check("wilson tau >= point-estimate tau (conservative)",
      routing.select_threshold(conf_n, corr_n, 0.9) >= point_tau - 1e-9)

print("[tokenizer]")
check("empty text -> placeholder token", len(jev.tokenize("")) == 1)
check("deterministic hashing", jev.tokenize("Hello World") == jev.tokenize("Hello World"))

print("[loss/grad sanity on odd batch]")
tr_t, tr_i, tr_u, tr_f = data.make_dataset(16, seed=9)
lbl = {"intent": tr_i, "urgency": tr_u - 1, "is_fraudulent": tr_f.astype(float)}
loss, per_q, grads = jev.losses_and_grads(model, tr_t, QUESTIONS, lbl, RNG)
allfinite = all(np.all(np.isfinite(g)) for g in grads.values())
check("loss finite + all gradients finite on mixed batch",
      np.isfinite(loss) and allfinite)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("ALL TESTS PASSED")
