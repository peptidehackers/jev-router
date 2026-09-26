"""SalesRLAgent-architecture training: the dataset's own 3072-dim embeddings
(frozen) -> trained encoder/head stack, same three primitives and Wilson gate.

This is the paper's input representation (arXiv 2503.23303), as opposed to the
hashed-text encoder in train_real.py which could not separate conversion
signal (base accuracy at chance despite strong confidence separation).

Usage: ./.venv/bin/python train_real_emb.py [emb_sample.npz]
"""
import hashlib
import json
import sys

import numpy as np

import calibration
import routing
from jev import Linear, Question, _h, q_sig, Q_VOCAB, softmax, sigmoid

TARGETS = [0.85, 0.90, 0.93, 0.95, 0.99]
LEVEL_CENTERS = [0.1, 0.3, 0.5, 0.7, 0.9]


class EmbModel:
    """Frozen 3072-dim embeddings -> Linear/tanh encoder -> shared mixer ->
    parallel typed heads. Explicit gradients (no framework)."""

    def __init__(self, rng, d_in=3072, d_hidden=128, d_q=32, d_head=64):
        self.rng = rng
        self.l1 = Linear(d_in, d_hidden, rng)
        self.lq = Linear(d_hidden + d_q, d_head, rng)
        self.Qe = rng.normal(0.0, 0.1, (Q_VOCAB, d_q))
        self.heads = {}
        self.d_in, self.d_hidden, self.d_q, self.d_head = d_in, d_hidden, d_q, d_head

    def _head(self, q):
        key = (q.name, q.kind)
        if key not in self.heads:
            if q.kind == "choice":
                self.heads[key] = Linear(self.d_head, q.n, self.rng)
            elif q.kind == "score":
                self.heads[key] = (Linear(self.d_head, 1, self.rng),
                                   Linear(self.d_head, q.n, self.rng))
            else:
                self.heads[key] = Linear(self.d_head, 1, self.rng)
        return self.heads[key]

    def forward(self, X, questions):
        z1 = self.l1.f(X)
        a1 = np.tanh(z1)
        N = X.shape[0]
        outs, caches = [], []
        for q in questions:
            qe = self.Qe[_h(q_sig(q), Q_VOCAB)]
            xq = np.concatenate([a1, np.broadcast_to(qe, (N, self.d_q))], axis=1)
            z2 = self.lq.f(xq)
            a2 = np.tanh(z2)
            if q.kind == "choice":
                logits = self._head(q).f(a2)
                p = softmax(logits)
                outs.append({"kind": "choice", "name": q.name, "probs": p,
                             "choice": np.argmax(p, 1), "confidence": p.max(1)})
                caches.append(("choice", q, X, z1, a1, xq, z2, a2, logits))
            elif q.kind == "score":
                vhead, dhead = self._head(q)
                v_raw = vhead.f(a2)[:, 0]
                v = np.clip(v_raw, float(min(q.levels)), float(max(q.levels)))
                dist = softmax(dhead.f(a2))
                outs.append({"kind": "score", "name": q.name, "value": v,
                             "distribution": dist, "levels": q.levels,
                             "confidence": dist.max(1)})
                caches.append(("score", q, X, z1, a1, xq, z2, a2, vhead, dhead, dist, v_raw))
            else:
                z = self._head(q).f(a2)[:, 0]
                p = sigmoid(z)
                outs.append({"kind": "noul", "name": q.name,
                             "probability": p, "is_true": p > 0.5})
                caches.append(("noul", q, X, z1, a1, xq, z2, a2, z))
        return outs, caches

    def params(self, questions):
        ps = {"l1.W": self.l1.W, "l1.b": self.l1.b,
              "lq.W": self.lq.W, "lq.b": self.lq.b, "Qe": self.Qe}
        for q in questions:
            h = self._head(q)
            if q.kind == "score":
                ps[f"h.{q.name}.v.W"], ps[f"h.{q.name}.v.b"] = h[0].W, h[0].b
                ps[f"h.{q.name}.d.W"], ps[f"h.{q.name}.d.b"] = h[1].W, h[1].b
            else:
                ps[f"h.{q.name}.W"], ps[f"h.{q.name}.b"] = h.W, h.b
        return ps


def losses_and_grads(m, X, questions, labels, mode="ce"):
    outs, caches = m.forward(X, questions)
    N = X.shape[0]
    ps = m.params(questions)
    grads = {k: np.zeros_like(v) for k, v in ps.items()}
    dl1_W = np.zeros_like(m.l1.W); dl1_b = np.zeros_like(m.l1.b)
    dlq_W = np.zeros_like(m.lq.W); dlq_b = np.zeros_like(m.lq.b)
    dQe = np.zeros_like(m.Qe)
    total = 0.0
    for o, c, q in zip(outs, caches, questions):
        kind, _, x, z1, a1, xq, z2, a2 = c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
        y = labels[q.name]
        da2 = np.zeros_like(a2)
        if kind == "choice":
            p = o["probs"]
            onehot = np.zeros_like(p); onehot[np.arange(N), y] = 1.0
            if mode == "ce":
                dlogits = (p - onehot) / N
                loss = -np.log(p[np.arange(N), y] + 1e-12).mean()
            else:
                s = ((p - onehot) * p).sum(1)
                dlogits = 2.0 * p * ((p - onehot) - s[:, None]) / N
                loss = ((p - onehot) ** 2).sum(1).mean()
            da2 += dlogits @ m.heads[(q.name, q.kind)].W
            grads[f"h.{q.name}.W"] += dlogits.T @ a2
            grads[f"h.{q.name}.b"] += dlogits.sum(0)
        elif kind == "score":
            vhead, dhead, dist, v_raw = c[8], c[9], c[10], c[11]
            onehot = np.zeros_like(dist); onehot[np.arange(N), y] = 1.0
            ddist = (dist - onehot) / N
            ce = -np.log(dist[np.arange(N), y] + 1e-12).mean()
            yv = np.asarray(q.levels, float)[y]
            dv = 2.0 * (v_raw - yv) / N
            mse = ((v_raw - yv) ** 2).mean()
            loss = ce + mse
            da2 += ddist @ dhead.W + dv[:, None] @ vhead.W
            grads[f"h.{q.name}.d.W"] += ddist.T @ a2
            grads[f"h.{q.name}.d.b"] += ddist.sum(0)
            grads[f"h.{q.name}.v.W"] += dv[:, None].T @ a2
            grads[f"h.{q.name}.v.b"] += dv.sum(0)
        else:
            p = o["probability"]
            if mode == "ce":
                dz = (p - y) / N
                loss = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean()
            else:
                dz = 2.0 * (p - y) * p * (1.0 - p) / N
                loss = ((p - y) ** 2).mean()
            da2 += dz[:, None] @ m.heads[(q.name, q.kind)].W
            grads[f"h.{q.name}.W"] += dz[:, None].T @ a2
            grads[f"h.{q.name}.b"] += dz.sum(0)
        total += loss
        dtanh = (1 - np.tanh(z2) ** 2) * da2
        dxq = dtanh @ m.lq.W
        dlq_W += dtanh.T @ xq
        dlq_b += dtanh.sum(0)
        da1 = dxq[:, :m.d_hidden]
        dQe[_h(q_sig(q), Q_VOCAB)] += dxq[:, m.d_hidden:].sum(0)
        dtanh1 = (1 - np.tanh(z1) ** 2) * da1
        dl1_W += dtanh1.T @ x
        dl1_b += dtanh1.sum(0)
    grads["l1.W"], grads["l1.b"] = dl1_W, dl1_b
    grads["lq.W"], grads["lq.b"] = dlq_W, dlq_b
    grads["Qe"] = dQe
    return total, grads


def grad_check(m, X, questions, labels, eps=1e-5, budget=150):
    rng = np.random.default_rng(7)
    an_vec, num_vec = [], []
    for name, p in m.params(questions).items():
        _, grads = losses_and_grads(m, X, questions, labels)
        an = grads[name]
        for k in rng.integers(0, an.size, size=min(budget // 6, an.size)):
            idx = np.unravel_index(int(k), an.shape)
            save = p[idx]
            p[idx] = save + eps
            lp = losses_and_grads(m, X, questions, labels)[0]
            p[idx] = save - eps
            lm = losses_and_grads(m, X, questions, labels)[0]
            p[idx] = save
            an_vec.append(an[idx])
            num_vec.append((lp - lm) / (2 * eps))
    a, n = np.asarray(an_vec), np.asarray(num_vec)
    return float(np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n) + 1e-12))


def split_of(company_id):
    b = int(hashlib.md5(company_id.encode()).hexdigest()[:4], 16) % 100
    return "train" if b < 70 else ("cal" if b < 85 else "test")


def eng_bin(x):
    return min(4, int(x / 0.2))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "emb_sample.npz"
    d = np.load(path, allow_pickle=False)
    X, comp = d["X"], d["company_id"]
    styles = sorted(set(d["style"][[split_of(c) == "train" for c in comp]]))
    print(f"rows={len(X)} companies={len(set(comp))} train-styles={len(styles)}")

    splits = {"train": [], "cal": [], "test": []}
    for i, c in enumerate(comp):
        splits[split_of(c)].append(i)
    for s in splits:
        print(f"  {s}: rows={len(splits[s])} companies={len(set(comp[splits[s]]))}")

    Q = [Question("conversion", "noul"),
         Question("style", "choice", options=styles),
         Question("engagement", "score", levels=LEVEL_CENTERS)]

    def pack(idxs):
        keep = [i for i in idxs if d["style"][i] in styles]
        y = {"conversion": d["outcome"][keep].astype(float),
             "style": np.array([styles.index(d["style"][i]) for i in keep]),
             "engagement": np.array([eng_bin(float(d["engagement"][i])) for i in keep])}
        return X[keep], y, keep

    Xtr, ytr, _ = pack(splits["train"])
    Xca, yca, ica = pack(splits["cal"])
    Xte, yte, ite = pack(splits["test"])
    print(f"usable: train={len(Xtr)} cal={len(Xca)} test={len(Xte)}")

    m = EmbModel(np.random.default_rng(0))
    print(f"gradient check (L2 ratio): {grad_check(m, Xtr[:8], Q, {k: v[:8] for k, v in ytr.items()}):.2e}")

    rng = np.random.default_rng(1)
    n = len(Xtr)
    for mode, epochs, lr in (("ce", 40, 0.1), ("brier", 10, 0.05)):
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s0 in range(0, n, 64):
                bi = idx[s0:s0 + 64]
                _, g = losses_and_grads(m, Xtr[bi], Q, {k: v[bi] for k, v in ytr.items()}, mode)
                ps = m.params(Q)
                for k, gg in g.items():
                    ps[k] -= lr * gg

    np.savez("model_emb.npz", **m.params(Q))

    def evaluate(Xs, ys, tag):
        outs, _ = m.forward(Xs, Q)
        by = {o["name"]: o for o in outs}
        p = by["conversion"]["probability"]
        corr = (p > 0.5) == (ys["conversion"] > 0.5)
        conf = np.maximum(p, 1 - p)
        scorr = by["style"]["choice"] == ys["style"]
        ecorr = by["engagement"]["distribution"].argmax(1) == ys["engagement"]
        print(f"\n[{tag}] conversion: acc={corr.mean():.3f} "
              f"brier={((p - ys['conversion']) ** 2).mean():.3f} "
              f"ECE={calibration.expected_calibration_error(conf, corr):.3f}")
        print(f"[{tag}] style:      acc={scorr.mean():.3f} "
              f"brier={calibration.brier(by['style']['probs'], ys['style']):.3f} "
              f"ECE={calibration.expected_calibration_error(by['style']['confidence'], scorr):.3f}")
        print(f"[{tag}] engagement: rubric_acc={ecorr.mean():.3f} "
              f"ECE={calibration.expected_calibration_error(by['engagement']['confidence'], ecorr):.3f}")
        return corr, conf, by

    corr_ca, conf_ca, by_ca = evaluate(Xca, yca, "CAL")
    corr_te, conf_te, by_te = evaluate(Xte, yte, "TEST")

    print("\n=== confidence-vs-correctness (conversion, TEST) ===")
    for f, acc in routing.risk_coverage_curve(conf_te, corr_te):
        print(f"   coverage {f:.0%} -> selective accuracy {acc:.1%}")
    k = len(conf_te) // 10
    hi = corr_te[np.argsort(-conf_te)[:k]].mean()
    lo = corr_te[np.argsort(-conf_te)[-k:]].mean()
    print(f"   top-decile {hi:.3f} vs bottom-decile {lo:.3f} (gap {hi - lo:+.3f})")

    print("\n=== Wilson-certified gate (fit CAL, eval TEST) ===")
    thr = {}
    for t in TARGETS:
        tau = routing.select_threshold(conf_ca, corr_ca, t)
        if tau is None:
            print(f"  target {t:.0%}: NOT CERTIFIABLE (cal n={len(Xca)}) -> escalate all")
            thr[str(t)] = None
            continue
        cov, acc = routing.coverage_at(conf_te, corr_te, tau)
        print(f"  target {t:.0%}: tau={tau:.3f}  TEST auto-act={cov:.1%} @ {acc:.1%}")
        thr[str(t)] = tau
    json.dump(thr, open("thresholds_emb.json", "w"), indent=1)

    np.savez("test_preds_emb.npz",
             conversation_id=d["conversation_id"][ite],
             p=by_te["conversion"]["probability"],
             pred=(by_te["conversion"]["probability"] > 0.5).astype(int),
             conf=np.maximum(by_te["conversion"]["probability"],
                             1 - by_te["conversion"]["probability"]),
             Xte=Xte)

    rows_out = [{"conversation_id": str(d["conversation_id"][i]),
                 "full_text": str(d["full_text"][i]),
                 "outcome": int(d["outcome"][i]),
                 "style": str(d["style"][i]),
                 "engagement": float(d["engagement"][i])} for i in ite]
    json.dump(rows_out, open("test_set_emb.jsonl", "w"))
    print(f"\nwrote model_emb.npz, thresholds_emb.json, test_set_emb.jsonl ({len(rows_out)})")


if __name__ == "__main__":
    main()
