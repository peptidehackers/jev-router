"""End-to-end demonstration of the Jev / "System One" reimplementation.

Pipeline: gradient-check -> CE training -> Brier (RLCD) stage -> temperature
scaling, reporting accuracy, calibration (ECE/Brier), latency, and a sample of
the type-safe structured output on every step.
"""
import json
import time

import numpy as np

import jev
from jev import Model, Question, softmax
import data
import calibration

RNG = np.random.default_rng(0)

QUESTIONS = [
    Question("intent", "choice", options=data.INTENTS),
    Question("urgency", "score", levels=[1, 2, 3, 4, 5]),
    Question("is_fraudulent", "noul"),
]


def build_labels(i, u, f):
    return {"intent": i, "urgency": u - 1, "is_fraudulent": f.astype(float)}


def evaluate(model, texts, labels):
    outs, _, _ = model.forward(texts, QUESTIONS)
    by_name = {o["name"]: o for o in outs}
    res = {}
    # intent (choice)
    c = by_name["intent"]
    pred = c["choice"]
    y = labels["intent"]
    res["intent_acc"] = (pred == y).mean()
    res["intent_ece"] = calibration.expected_calibration_error(c["confidence"], pred == y)
    res["intent_brier"] = calibration.brier(c["probs"], y)
    # fraud (noul)
    n = by_name["is_fraudulent"]
    yf = labels["is_fraudulent"]
    pf = n["is_true"].astype(float)
    res["fraud_acc"] = (pf == yf).mean()
    res["fraud_ece"] = calibration.expected_calibration_error(n["probability"], pf == yf)
    # urgency (score)
    s = by_name["urgency"]
    lvl = s["distribution"].argmax(1)
    res["urgency_acc"] = (lvl == labels["urgency"]).mean()
    levels = np.asarray(QUESTIONS[1].levels, float)
    expected = (s["distribution"] * levels).sum(1)
    res["urgency_mae"] = np.abs(expected - levels[labels["urgency"]]).mean()
    return res, by_name


def train(model, texts, labels, mode, epochs, lr, batch=128):
    n = len(texts)
    for _ in range(epochs):
        idx = RNG.permutation(n)
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            bt = [texts[i] for i in bi]
            bl = {k: v[bi] for k, v in labels.items()}
            _, _, grads = jev.losses_and_grads(model, bt, QUESTIONS, bl, RNG, mode=mode)
            ps = model.params(QUESTIONS)
            for k, g in grads.items():
                ps[k] -= lr * g


def gradient_check(model, texts, labels, eps=1e-5, budget=200):
    """Finite-difference check via aggregate L2 ratio (robust to sparse grads).
    Returns a single relative error; <1e-5 means gradients are correct."""
    ps = model.params(QUESTIONS)
    rng = np.random.default_rng(7)
    an_vec, num_vec = [], []
    for name, p in ps.items():
        _, _, grads = jev.losses_and_grads(model, texts, QUESTIONS, labels, RNG)
        an = grads[name]
        n = an.size
        picks = rng.integers(0, n, size=min(budget, n))
        for k in picks:
            idx = np.unravel_index(int(k), an.shape)
            save = p[idx]
            p[idx] = save + eps
            lp = jev.losses_and_grads(model, texts, QUESTIONS, labels, RNG)[0]
            p[idx] = save - eps
            lm = jev.losses_and_grads(model, texts, QUESTIONS, labels, RNG)[0]
            p[idx] = save
            an_vec.append(an[idx])
            num_vec.append((lp - lm) / (2 * eps))
    a = np.asarray(an_vec)
    n = np.asarray(num_vec)
    return float(np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n) + 1e-12))


def structured_output(model, text):
    outs, _, _ = model.forward([text], QUESTIONS)
    d = {}
    for o in outs:
        if o["kind"] == "choice":
            d[o["name"]] = {
                "choice": QUESTIONS[0].options[int(o["choice"][0])],
                "probabilities": {k: round(v, 3) for k, v in zip(QUESTIONS[0].options, o["probs"][0])},
                "confidence": round(float(o["confidence"][0]), 3),
            }
        elif o["kind"] == "score":
            levels = np.asarray(QUESTIONS[1].levels, float)
            d[o["name"]] = {
                "value": round(float(o["value"][0]), 3),
                "distribution": {int(l): round(float(v), 3) for l, v in zip(levels, o["distribution"][0])},
                "confidence": round(float(o["confidence"][0]), 3),
            }
        else:
            d[o["name"]] = {
                "is_true": bool(o["is_true"][0]),
                "probability": round(float(o["probability"][0]), 3),
            }
    return d


def main():
    tr_t, tr_i, tr_u, tr_f = data.make_dataset(3000, seed=1)
    ca_t, ca_i, ca_u, ca_f = data.make_dataset(400, seed=2)
    ev_t, ev_i, ev_u, ev_f = data.make_dataset(500, seed=3)
    tr_l = build_labels(tr_i, tr_u, tr_f)
    ca_l = build_labels(ca_i, ca_u, ca_f)
    ev_l = build_labels(ev_i, ev_u, ev_f)

    model = Model(RNG, d_emb=64, d_hidden=128, d_q=32, d_head=64)

    # 1. gradient check
    gc_t = [tr_t[i] for i in range(8)]
    gc_l = {k: v[:8] for k, v in tr_l.items()}
    print(f"gradient check (L2 ratio, want <1e-5): {gradient_check(model, gc_t, gc_l):.2e}")

    # 2. CE training
    train(model, tr_t, tr_l, mode="ce", epochs=40, lr=0.1)
    r, by = evaluate(model, ev_t, ev_l)
    print("\n[after CE/NLL]  intent_acc=%.3f intent_ECE=%.3f intent_brier=%.3f  fraud_acc=%.3f  urgency_acc=%.3f"
          % (r["intent_acc"], r["intent_ece"], r["intent_brier"], r["fraud_acc"], r["urgency_acc"]))

    # 3. Brier (RLCD) stage
    train(model, tr_t, tr_l, mode="brier", epochs=10, lr=0.05)
    r, by = evaluate(model, ev_t, ev_l)
    print("[after Brier]  intent_acc=%.3f intent_ECE=%.3f intent_brier=%.3f  fraud_acc=%.3f"
          % (r["intent_acc"], r["intent_ece"], r["intent_brier"], r["fraud_acc"]))
    ece_before = r["intent_ece"]

    # 4. temperature scaling on choice head
    _, by = evaluate(model, ca_t, ca_l)
    Z = np.log(by["intent"]["probs"] + 1e-12)  # recover logits (shift-invariant)
    T, nll = calibration.fit_temperature(Z, ca_l["intent"])
    print(f"[temperature] fitted T={T:.3f} (cal NLL {nll:.3f})")

    # re-evaluate with temperature applied to choice head
    outs, _, _ = model.forward(ev_t, QUESTIONS)
    c = {o["name"]: o for o in outs}["intent"]
    probs_T = softmax(np.log(c["probs"] + 1e-12) / T)
    pred_T = probs_T.argmax(1)
    ece_after = calibration.expected_calibration_error(probs_T.max(1), pred_T == ev_l["intent"])
    acc_after = (pred_T == ev_l["intent"]).mean()
    brier_after = calibration.brier(probs_T, ev_l["intent"])
    print(f"[after temp]  intent_acc=%.3f intent_ECE=%.3f intent_brier=%.3f   (ECE %+.3f vs pre-temp)"
          % (acc_after, ece_after, brier_after, ece_after - ece_before))

    # reliability table — the RLCD property made visible: confidence == accuracy
    centers, accs, cnts = calibration.reliability(probs_T.max(1), pred_T == ev_l["intent"])
    print("\nreliability (predicted confidence vs empirical accuracy):")
    for c, a, n in zip(centers, accs, cnts):
        if n:
            print(f"   conf~{c:.2f}  n={n:3d}  accuracy={a:.2f}")

    # 5. latency: single forward pass (non-autoregressive)
    t0 = time.perf_counter()
    for _ in range(100):
        model.forward([ev_t[0]], QUESTIONS)
    per_call = (time.perf_counter() - t0) / 100
    t0 = time.perf_counter()
    model.forward(ev_t[:128], QUESTIONS)
    batch_128 = time.perf_counter() - t0
    print(f"\nlatency: {per_call*1000:.2f} ms single sample, {batch_128*1000:.2f} ms for 128 (parallel pass)")

    # 6. sample structured outputs
    print("\nsample structured outputs (type-safe schema):")
    for ex in ["payment invoice charge is overdue urgent asap", "production down emergency data loss"]:
        print(json.dumps(structured_output(model, ex), indent=1))


if __name__ == "__main__":
    main()
