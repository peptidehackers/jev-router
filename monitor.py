"""Calibration & distribution-drift monitor. Read-only evidence producer.

Input: an npz with the same contract as emb_sample.npz
  (X float[n,3072], outcome int[n], company_id str[n]) — production outcome
  labels can be ingested through this exact format without architectural change.

Measures: ECE, Brier, reliability, class balance, confidence distribution,
accuracy, risk-vs-coverage, and Wilson-certified coverage at 85/90/93/95/99%.
Compares each against the baseline recorded at approval time (baseline.json).
NEVER retrains, NEVER loosens thresholds. If drift is flagged the operator must
re-approve (approve.py); until then infer.py keeps failing closed.
"""
import json
import sys

import numpy as np

import calibration
import routing
from jev import Question
from train_real_emb import EmbModel

TARGETS = [0.85, 0.90, 0.93, 0.95, 0.99]


def load_head():
    m = EmbModel(np.random.default_rng(0))
    Q = [Question("conversion", "noul")]
    ps = m.params(Q)
    saved = np.load("model_emb.npz")
    for k in ps:
        ps[k][:] = saved[k]
    return m, Q


def main(path):
    baseline = json.load(open("baseline.json"))
    d = np.load(path)
    X, y = d["X"], d["outcome"].astype(float)
    m, Q = load_head()
    outs, _ = m.forward(X, Q)
    p = outs[0]["probability"]
    conf = np.maximum(p, 1 - p)
    corr = (p > 0.5) == (y > 0.5)

    cur = {
        "n": int(len(y)),
        "accuracy": float(corr.mean()),
        "brier": float(((p - y) ** 2).mean()),
        "ece": calibration.expected_calibration_error(conf, corr),
        "class_balance": float((y > 0.5).mean()),
        "confidence_mean": float(conf.mean()),
        "confidence_deciles": [float(q) for q in np.quantile(conf, np.linspace(0.1, 1.0, 10))],
    }
    b = baseline["cal"]
    lim = baseline["drift_limits"]

    flags = []
    def flag(name, cond, detail):
        flags.append(("DRIFT" if cond else "ok", name, detail))
    flag("accuracy_drop", cur["accuracy"] < b["accuracy"] - lim["accuracy_drop"],
         f"{cur['accuracy']:.3f} vs baseline {b['accuracy']:.3f} (limit -{lim['accuracy_drop']})")
    flag("brier_increase", cur["brier"] > b["brier"] + lim["brier_increase"],
         f"{cur['brier']:.3f} vs {b['brier']:.3f} (+{lim['brier_increase']})")
    flag("ece_increase", cur["ece"] > b["ece"] + lim["ece_increase"],
         f"{cur['ece']:.3f} vs {b['ece']:.3f} (+{lim['ece_increase']})")
    flag("class_balance_shift", abs(cur["class_balance"] - b["class_balance"]) > lim["class_balance_shift"],
         f"{cur['class_balance']:.3f} vs {b['class_balance']:.3f}")
    flag("confidence_mean_shift", abs(cur["confidence_mean"] - b["confidence_mean"]) > lim["confidence_mean_shift"],
         f"{cur['confidence_mean']:.3f} vs {b['confidence_mean']:.3f}")

    print(f"monitor: n={cur['n']}  acc={cur['accuracy']:.3f}  brier={cur['brier']:.3f}  "
          f"ECE={cur['ece']:.3f}  balance={cur['class_balance']:.3f}  "
          f"conf_mean={cur['confidence_mean']:.3f}")
    print("\nreliability (confidence -> empirical accuracy):")
    centers, accs, cnts = calibration.reliability(conf, corr)
    for c, a, n in zip(centers, accs, cnts):
        if n:
            print(f"   {c:.2f}  n={n:4d}  acc={a:.2f}")
    print("\nrisk-coverage:")
    for f, a in routing.risk_coverage_curve(conf, corr):
        print(f"   {f:.0%} -> {a:.1%}")

    print("\ncertified-gate simulation on observed data (reference only):")
    approved = json.load(open("approved.json"))
    for t in TARGETS:
        e = approved["targets"][f"{t:.2f}"]
        if e["tau"] is None:
            print(f"   {t:.0%}: uncertified (mandatory escalation)")
            continue
        cov, acc = routing.coverage_at(conf, corr, e["tau"])
        print(f"   {t:.0%}: tau={e['tau']:.3f}  observed auto-act={cov:.1%} @ {acc:.1%}")

    print("\ndrift checks:")
    any_drift = False
    for status, name, detail in flags:
        if status == "DRIFT":
            any_drift = True
        print(f"   [{status}] {name}: {detail}")
    if any_drift:
        print("\nVERDICT: DRIFT DETECTED. Thresholds remain unchanged; production "
              "must fail closed (escalate) until re-approved via approve.py. "
              "No automatic retraining is performed.")
        return 1
    print("\nVERDICT: within baseline limits.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "emb_sample.npz"))
