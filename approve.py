"""Threshold approval: turns calibration evidence into a versioned, fingerprinted
artifact that production inference will refuse to run without.

Writes:
  approved.json  — per-target Wilson-certified thresholds + fingerprints
                   (model sha256, dataset sha256, split assignment hash,
                   cal sample size, Wilson z, creation time, approval metrics)
  baseline.json  — training/calibration baselines the drift monitor compares
                   against (never auto-retrains, never loosens)

A threshold is only valid for the exact model+data+split it was fitted on;
infer.py verifies all three fingerprints before acting.
"""
import datetime
import hashlib
import json

import numpy as np

import calibration
import routing
from jev import Question
from train_real_emb import EmbModel, split_of

TARGETS = [0.85, 0.90, 0.93, 0.95, 0.99]
Z = 1.96  # Wilson 95%


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_head():
    m = EmbModel(np.random.default_rng(0))
    Q = [Question("conversion", "noul")]
    ps = m.params(Q)
    saved = np.load("model_emb.npz")
    for k in ps:
        ps[k][:] = saved[k]
    return m, Q


def cal_evidence():
    d = np.load("emb_sample.npz")
    idx = [i for i, c in enumerate(d["company_id"]) if split_of(str(c)) == "cal"]
    X = d["X"][idx]
    y = d["outcome"][idx].astype(float)
    m, Q = load_head()
    outs, _ = m.forward(X, Q)
    p = outs[0]["probability"]
    conf = np.maximum(p, 1 - p)
    corr = (p > 0.5) == (y > 0.5)
    return d, idx, y, p, conf, corr, m, Q


def main():
    d, idx, y, p, conf, corr, m, Q = cal_evidence()

    # split fingerprint: company -> split assignment, hashed
    assignment = {str(c): split_of(str(c)) for c in sorted(set(d["company_id"]), key=str)}
    split_fp = hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()[:16]

    targets_out = {}
    for t in TARGETS:
        tau = routing.select_threshold(conf, corr, t)
        entry = {"target": t, "wilson_z": Z, "tau": tau,
                 "cal_n": int(len(conf)),
                 "certified": tau is not None}
        if tau is not None:
            cov, acc = routing.coverage_at(conf, corr, tau)
            entry["cal_coverage"] = round(cov, 4)
            entry["cal_accuracy"] = round(acc, 4)
        else:
            entry["reason"] = (f"Wilson {Z}-lower-bound cannot certify {t:.0%} "
                               f"on n={len(conf)}; escalation is mandatory")
        targets_out[f"{t:.2f}"] = entry

    # test-set reference metrics (evidence, not approval basis)
    t_idx = [i for i, c in enumerate(d["company_id"]) if split_of(str(c)) == "test"]
    Xt = d["X"][t_idx]
    yt = d["outcome"][t_idx].astype(float)
    outs_t, _ = m.forward(Xt, Q)
    pt = outs_t[0]["probability"]
    conf_t = np.maximum(pt, 1 - pt)
    corr_t = (pt > 0.5) == (yt > 0.5)

    approved = {
        "artifact": "jev-router-thresholds",
        "schema_version": 1,
        "decision": "conversion(noul)",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_sha256": sha256_file("model_emb.npz"),
        "dataset_sha256": sha256_file("emb_sample.npz"),
        "split_fingerprint": split_fp,
        "split_rule": "md5(company_id)[:4]%100 <70 train <85 cal else test",
        "targets": targets_out,
        "approval_metrics": {
            "cal_n": int(len(conf)),
            "cal_accuracy": round(float(corr.mean()), 4),
            "cal_brier": round(float(((p - y) ** 2).mean()), 4),
            "cal_ece": round(calibration.expected_calibration_error(conf, corr), 4),
        },
    }
    json.dump(approved, open("approved.json", "w"), indent=1)

    baseline = {
        "created_at": approved["created_at"],
        "model_sha256": approved["model_sha256"],
        "dataset_sha256": approved["dataset_sha256"],
        "split_fingerprint": split_fp,
        "cal": {
            "n": int(len(conf)),
            "accuracy": float(corr.mean()),
            "brier": float(((p - y) ** 2).mean()),
            "ece": calibration.expected_calibration_error(conf, corr),
            "class_balance": float((y > 0.5).mean()),
            "confidence_mean": float(conf.mean()),
            "confidence_deciles": [float(q) for q in np.quantile(conf, np.linspace(0.1, 1.0, 10))],
            "reliability": calibration.reliability(conf, corr),
            "risk_coverage": routing.risk_coverage_curve(conf, corr),
        },
        "test": {
            "n": int(len(conf_t)),
            "accuracy": float(corr_t.mean()),
            "brier": float(((pt - yt) ** 2).mean()),
            "ece": calibration.expected_calibration_error(conf_t, corr_t),
            "class_balance": float((yt > 0.5).mean()),
            "confidence_mean": float(conf_t.mean()),
        },
        "drift_limits": {  # flag thresholds for monitor.py; never auto-action
            "accuracy_drop": 0.05, "brier_increase": 0.05, "ece_increase": 0.05,
            "class_balance_shift": 0.10, "confidence_mean_shift": 0.10,
        },
    }
    json.dump(baseline, open("baseline.json", "w"), indent=1)
    print(f"approved.json + baseline.json written "
          f"(model {approved['model_sha256'][:12]}…, cal n={len(conf)})")
    for t, e in targets_out.items():
        print(f"  {float(t):.0%}: " + (f"tau={e['tau']:.3f} cal-cov={e['cal_coverage']:.1%} "
                                       f"@ {e['cal_accuracy']:.1%}" if e["certified"]
                                       else "NOT CERTIFIED -> mandatory escalation"))


if __name__ == "__main__":
    main()
