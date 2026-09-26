"""Real-data pipeline: load -> group-split -> train -> evaluate -> gate.

Splits are grouped by company_id (hash-based, deterministic) so no company's
conversations straddle splits — prevents customer leakage. Label spaces
(styles) are discovered from TRAIN only; unseen values in cal/test are counted
and excluded from that question's metrics rather than silently mapped.

Usage: ./.venv/bin/python train_real.py [real_data.jsonl]
Writes: model.npz, thresholds.json, test_set.jsonl, cal_set.jsonl
"""
import hashlib
import json
import sys

import numpy as np

import calibration
import jev
import routing
from jev import Model, Question

TARGETS = [0.85, 0.90, 0.93, 0.95, 0.99]
LEVEL_CENTERS = [0.1, 0.3, 0.5, 0.7, 0.9]  # engagement rubric (bin centers)


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def split_of(company_id):
    b = int(hashlib.md5(company_id.encode()).hexdigest()[:4], 16) % 100
    return "train" if b < 70 else ("cal" if b < 85 else "test")


def prep(rows):
    splits = {"train": [], "cal": [], "test": []}
    companies = {"train": set(), "cal": set(), "test": set()}
    for r in rows:
        s = split_of(r["company_id"])
        splits[s].append(r)
        companies[s].add(r["company_id"])
    return splits, companies


def engagement_bin(x):
    return min(4, int(x / 0.2))  # 0.0-1.0 -> 0..4 (bin index)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "real_data.jsonl"
    rows = load(path)
    splits, companies = prep(rows)
    print(f"rows: train={len(splits['train'])} cal={len(splits['cal'])} "
          f"test={len(splits['test'])}")
    print(f"companies (leakage check, must be disjoint): "
          f"train={len(companies['train'])} cal={len(companies['cal'])} "
          f"test={len(companies['test'])} "
          f"overlap={len(companies['train'] & companies['cal']) + len(companies['train'] & companies['test']) + len(companies['cal'] & companies['test'])}")

    # label spaces from TRAIN only
    styles = sorted({r["conversation_style"] for r in splits["train"]})
    for s in ("cal", "test"):
        unseen = sum(1 for r in splits[s] if r["conversation_style"] not in styles)
        print(f"  unseen styles in {s}: {unseen}")
    rate = {s: np.mean([r["outcome"] for r in splits[s]]) for s in splits}
    print(f"  outcome rate (shift check): " + " ".join(f"{s}={rate[s]:.3f}" for s in splits))
    eng = {s: np.mean([r["customer_engagement"] for r in splits[s]]) for s in splits}
    print(f"  mean engagement (shift check): " + " ".join(f"{s}={eng[s]:.3f}" for s in splits))

    Q = [
        Question("conversion", "noul"),
        Question("style", "choice", options=styles),
        Question("engagement", "score", levels=LEVEL_CENTERS),
    ]

    def to_xy(split, require_all=True):
        texts, y_out, y_sty, y_eng, keep_ids = [], [], [], [], []
        skipped = 0
        for r in split:
            if r["conversation_style"] not in styles and require_all:
                skipped += 1
                continue
            texts.append(r["full_text"])
            y_out.append(float(r["outcome"]))
            y_sty.append(styles.index(r["conversation_style"]) if r["conversation_style"] in styles else -1)
            y_eng.append(engagement_bin(r["customer_engagement"]))
            keep_ids.append(r["conversation_id"])
        return texts, np.array(y_out), np.array(y_sty), np.array(y_eng), keep_ids, skipped

    tr_t, tr_o, tr_s, tr_e, _, sk = to_xy(splits["train"])
    ca_t, ca_o, ca_s, ca_e, ca_ids, sk_ca = to_xy(splits["cal"])
    te_t, te_o, te_s, te_e, te_ids, sk_te = to_xy(splits["test"])
    print(f"usable: train={len(tr_t)} cal={len(ca_t)} (skipped {sk_ca}) test={len(te_t)} (skipped {sk_te})")

    labels_tr = {"conversion": tr_o, "style": tr_s, "engagement": tr_e}
    labels_ca = {"conversion": ca_o, "style": ca_s, "engagement": ca_e}
    labels_te = {"conversion": te_o, "style": te_s, "engagement": te_e}

    model = Model(np.random.default_rng(0), d_emb=64, d_hidden=128, d_q=32, d_head=64)

    rng = np.random.default_rng(1)
    n = len(tr_t)
    for mode, epochs, lr in (("ce", 40, 0.1), ("brier", 10, 0.05)):
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s0 in range(0, n, 128):
                bi = idx[s0:s0 + 128]
                _, _, g = jev.losses_and_grads(
                    model, [tr_t[i] for i in bi], Q,
                    {k: v[bi] for k, v in labels_tr.items()}, rng, mode=mode)
                ps = model.params(Q)
                for k, gg in g.items():
                    ps[k] -= lr * gg

    # persist model
    ps = model.params(Q)
    np.savez("model.npz", **{k: v for k, v in ps.items()},
             **{f"meta_{k}": [v] for k, v in [("d_emb", model.d_emb), ("d_hidden", model.d_hidden),
                                               ("d_q", model.d_q), ("d_head", model.d_head)]})

    def eval_split(texts, labels, tag):
        outs, _, _ = model.forward(texts, Q)
        by = {o["name"]: o for o in outs}
        report = {}
        # conversion (noul) — the core decision
        p = by["conversion"]["probability"]
        pred = p > 0.5
        corr = pred == (labels["conversion"] > 0.5)
        conf = np.maximum(p, 1 - p)
        report["conversion"] = {
            "acc": float(corr.mean()),
            "brier": float(((p - labels["conversion"]) ** 2).mean()),
            "ece": calibration.expected_calibration_error(conf, corr),
            "pos_rate": float((labels["conversion"] > 0.5).mean()),
        }
        # style (choice)
        m = labels["style"] >= 0
        scorr = by["style"]["choice"][m] == labels["style"][m]
        report["style"] = {
            "acc": float(scorr.mean()),
            "brier": calibration.brier(by["style"]["probs"][m], labels["style"][m]),
            "ece": calibration.expected_calibration_error(by["style"]["confidence"][m], scorr),
            "n": int(m.sum()),
        }
        # engagement (score)
        ecorr = by["engagement"]["distribution"].argmax(1) == labels["engagement"]
        levels = np.asarray(LEVEL_CENTERS)
        exp_v = (by["engagement"]["distribution"] * levels).sum(1)
        report["engagement"] = {
            "rubric_acc": float(ecorr.mean()),
            "mae": float(np.abs(exp_v - levels[labels["engagement"]]).mean()),
            "ece": calibration.expected_calibration_error(by["engagement"]["confidence"], ecorr),
        }
        print(f"\n[{tag}] conversion: acc={report['conversion']['acc']:.3f} "
              f"brier={report['conversion']['brier']:.3f} ECE={report['conversion']['ece']:.3f}")
        print(f"[{tag}] style:      acc={report['style']['acc']:.3f} "
              f"brier={report['style']['brier']:.3f} ECE={report['style']['ece']:.3f} (n={report['style']['n']})")
        print(f"[{tag}] engagement: rubric_acc={report['engagement']['rubric_acc']:.3f} "
              f"mae={report['engagement']['mae']:.3f} ECE={report['engagement']['ece']:.3f}")
        return report, by, corr, conf

    rep_ca, by_ca, corr_ca, conf_ca = eval_split(ca_t, labels_ca, "CAL")
    rep_te, by_te, corr_te, conf_te = eval_split(te_t, labels_te, "TEST")

    # ---- critical gate: does confidence separate correct from incorrect? ----
    print("\n=== confidence-vs-correctness (conversion, TEST) ===")
    rc = routing.risk_coverage_curve(conf_te, corr_te)
    for f, acc in rc:
        print(f"   coverage {f:.0%} -> selective accuracy {acc:.1%}")
    top = np.sort(conf_te)[::-1]
    hi = corr_te[np.argsort(-conf_te)[:len(te_t) // 10]].mean()
    lo = corr_te[np.argsort(-conf_te)[-len(te_t) // 10:]].mean()
    print(f"   top-decile accuracy {hi:.3f} vs bottom-decile {lo:.3f} "
          f"(separation gap {hi - lo:+.3f})")

    # ---- Wilson-certified coverage sweep ----
    print("\n=== Wilson-certified gate (fit on CAL, evaluated on TEST) ===")
    thr = {}
    for t in TARGETS:
        tau = routing.select_threshold(conf_ca, corr_ca, t)
        if tau is None:
            print(f"  target {t:.0%}: NOT CERTIFIABLE on cal (n={len(ca_t)}) -> escalate all")
            thr[str(t)] = None
            continue
        cov, acc = routing.coverage_at(conf_te, corr_te, tau)
        print(f"  target {t:.0%}: tau={tau:.3f}  TEST auto-act={cov:.1%} @ {acc:.1%}")
        thr[str(t)] = tau
    json.dump(thr, open("thresholds.json", "w"), indent=1)

    # export held-out sets for the OMP three-path comparison
    for name, texts, ids, outs_labels, split_rows in (
            ("test", te_t, te_ids, labels_te, splits["test"]),
            ("cal", ca_t, ca_ids, labels_ca, splits["cal"])):
        rows_out = []
        it = iter(split_rows)
        for t_, id_ in zip(texts, ids):
            r = next(it)
            rows_out.append({"conversation_id": id_, "full_text": t_,
                             "outcome": int(r["outcome"]),
                             "style": r["conversation_style"],
                             "engagement": r["customer_engagement"]})
        json.dump(rows_out, open(f"{name}_set.jsonl", "w"), indent=0)
    print(f"\nwrote model.npz, thresholds.json, test_set.jsonl ({len(te_t)}), cal_set.jsonl ({len(ca_t)})")


if __name__ == "__main__":
    main()
