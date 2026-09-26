"""Jev / "System One" architecture — a non-autoregressive, schema-driven decision model.

One parallel forward pass over input state -> type-safe, calibrated probability
outputs for a fixed JSON schema built from three primitives:

  Choice  : discrete options -> full probability spread + argmax + confidence
  Score   : ordered rubric    -> continuous value + level distribution + confidence
  Noul    : boolean proposition -> P(true) + is_true

Numpy-only, explicit gradients (no framework). The design mirrors the contract of
TypeSafe AI's Jev: no text generation, no sequential decoding, outputs map
directly onto a pre-enumerated schema, and probabilities are trained to be
*calibrated* (a 0.80 confidence is correct ~80% of the time).
"""
import hashlib
import re

import numpy as np

VOCAB = 4096
Q_VOCAB = 256


def _h(s: str, m: int) -> int:
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "little") % m


def tokenize(text: str):
    """Deterministic word hashing (no vocab file, reproducible). Word-level
    features carry the decision signal; char trigrams were dropped because they
    dilute it ~3x."""
    t = text.lower()
    ids = [_h("w:" + w, VOCAB) for w in re.findall(r"[a-z0-9]+", t)]
    return ids or [_h("empty", VOCAB)]


# --------------------------------------------------------------------------- #
# Layers (explicit forward/backward)
# --------------------------------------------------------------------------- #
class Linear:
    def __init__(self, d_in, d_out, rng):
        self.W = rng.normal(0.0, 0.1, (d_out, d_in))
        self.b = np.zeros(d_out)

    def f(self, x):
        self._x = x
        return x @ self.W.T + self.b

    def backward(self, dy):
        return dy @ self.W, dy.T @ self._x, dy.sum(0)  # dx, dW, db


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class Question:
    def __init__(self, name, kind, options=None, levels=None):
        assert kind in ("choice", "score", "noul")
        if kind == "choice":
            assert options
            self.options = list(options)
        if kind == "score":
            assert levels
            self.levels = list(levels)
        self.name = name
        self.kind = kind
        self.n = len(options) if kind == "choice" else (len(levels) if kind == "score" else 1)


def q_sig(q: Question) -> str:
    extra = getattr(q, "options", None) or getattr(q, "levels", None) or []
    return q.name + q.kind + ",".join(str(x) for x in extra)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Model:
    """Encoder -> shared question mixer -> parallel typed heads."""

    def __init__(self, rng, d_emb=32, d_hidden=48, d_q=24, d_head=32):
        self.rng = rng
        self.E = rng.normal(0.0, 0.05, (VOCAB, d_emb))
        self.Qe = rng.normal(0.0, 0.1, (Q_VOCAB, d_q))
        self.l1 = Linear(d_emb, d_hidden, rng)          # encoder
        self.lq = Linear(d_hidden + d_q, d_head, rng)   # shared question mixer
        # per-question heads, keyed by (name, kind)
        self.heads = {}
        self.d_emb, self.d_hidden, self.d_q, self.d_head = d_emb, d_hidden, d_q, d_head

    def _head(self, q: Question, rng=None):
        key = (q.name, q.kind)
        if key not in self.heads:
            if q.kind == "choice":
                self.heads[key] = Linear(self.d_head, q.n, self.rng)
            elif q.kind == "score":
                self.heads[key] = (Linear(self.d_head, 1, self.rng), Linear(self.d_head, q.n, self.rng))
            else:
                self.heads[key] = Linear(self.d_head, 1, self.rng)
        return self.heads[key]

    def forward(self, texts, questions):
        """One parallel pass. Returns list of per-question output dicts."""
        tokens = [tokenize(t) for t in texts]
        N = len(texts)
        x = np.zeros((N, self.d_emb))
        for i, toks in enumerate(tokens):
            x[i] = self.E[toks].mean(0)

        z1 = self.l1.f(x)
        a1 = np.tanh(z1)  # (N, d_hidden) = H, the shared state encoding

        outs = []
        caches = []
        for q in questions:
            qe = self.Qe[_h(q_sig(q), Q_VOCAB)]
            xq = np.concatenate([a1, np.broadcast_to(qe, (N, self.d_q))], axis=1)
            z2 = self.lq.f(xq)
            a2 = np.tanh(z2)  # (N, d_head)

            if q.kind == "choice":
                logits = self._head(q, None).f(a2)
                p = softmax(logits)
                conf = p.max(1)
                outs.append({"kind": "choice", "name": q.name,
                             "probs": p, "choice": np.argmax(p, 1), "confidence": conf})
                caches.append(("choice", q, x, z1, a1, xq, z2, a2, logits))
            elif q.kind == "score":
                vhead, dhead = self._head(q, None)
                v_raw = vhead.f(a2)[:, 0]
                # type-safety: the reported value is clamped to the rubric
                # bounds (a Score answer can never escape its own schema);
                # training differentiates the raw value so gradients stay exact.
                v = np.clip(v_raw, float(min(q.levels)), float(max(q.levels)))
                dist = softmax(dhead.f(a2))      # distribution over rubric levels
                conf = dist.max(1)
                outs.append({"kind": "score", "name": q.name, "value": v,
                             "distribution": dist, "levels": q.levels, "confidence": conf})
                caches.append(("score", q, x, z1, a1, xq, z2, a2, vhead, dhead, dist, v_raw))
            else:
                z = self._head(q, None).f(a2)[:, 0]
                p = sigmoid(z)
                outs.append({"kind": "noul", "name": q.name,
                             "probability": p, "is_true": p > 0.5})
                caches.append(("noul", q, x, z1, a1, xq, z2, a2, z))
        return outs, tokens, caches

    def params(self, questions):
        """All trainable (param, gradient) pairs, in a stable order."""
        ps = {"E": self.E, "l1.W": self.l1.W, "l1.b": self.l1.b,
              "lq.W": self.lq.W, "lq.b": self.lq.b, "Qe": self.Qe}
        for q in questions:
            h = self._head(q, None)
            if q.kind == "score":
                ps[f"h.{q.name}.v.W"], ps[f"h.{q.name}.v.b"] = h[0].W, h[0].b
                ps[f"h.{q.name}.d.W"], ps[f"h.{q.name}.d.b"] = h[1].W, h[1].b
            else:
                ps[f"h.{q.name}.W"], ps[f"h.{q.name}.b"] = h.W, h.b
        return ps


# --------------------------------------------------------------------------- #
# Loss + gradients
# --------------------------------------------------------------------------- #
def losses_and_grads(model, texts, questions, labels, rng, mode="ce"):
    """labels: dict name -> targets. mode "ce" (NLL, proper scoring) or "brier"
    (RLCD-style: directly reward calibrated distributions via the Brier proper
    scoring rule, which RLHF-style reward maximization cannot fake with
    overconfidence). Returns (total_loss, per-question losses, grad dict)."""
    outs, tokens, caches = model.forward(texts, questions)
    N = len(texts)
    ps = model.params(questions)
    grads = {k: np.zeros_like(v) for k, v in ps.items()}

    dE = np.zeros_like(model.E)
    dl1_W = np.zeros_like(model.l1.W); dl1_b = np.zeros_like(model.l1.b)
    dlq_W = np.zeros_like(model.lq.W); dlq_b = np.zeros_like(model.lq.b)
    dQe = np.zeros_like(model.Qe)

    total = 0.0
    per_q = {}

    # tokens -> embedding scatter factors (for mean-pool backward)
    emb_scatter = []  # list of (idx, toks, 1/count)
    for toks in tokens:
        emb_scatter.append((toks, 1.0 / len(toks)))

    for o, c, q in zip(outs, caches, questions):
        kind, _, x, z1, a1, xq, z2, a2 = c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]
        y = labels[q.name]
        da2 = np.zeros_like(a2)
        if kind == "choice":
            logits = c[8]
            p = o["probs"]
            onehot = np.zeros_like(logits); onehot[np.arange(N), y] = 1.0
            if mode == "ce":
                dlogits = (p - onehot) / N
                loss = -np.log(p[np.arange(N), y] + 1e-12).mean()
            else:  # brier
                s = ((p - onehot) * p).sum(1)
                dlogits = 2.0 * p * ((p - onehot) - s[:, None]) / N
                loss = ((p - onehot) ** 2).sum(1).mean()
            da2 += dlogits @ model.heads[(q.name, q.kind)].W
            grads[f"h.{q.name}.W"] += dlogits.T @ a2
            grads[f"h.{q.name}.b"] += dlogits.sum(0)
        elif kind == "score":
            vhead, dhead = c[8], c[9]
            dist = c[10]
            v_raw = c[11]  # unclamped; loss/grad use raw so backprop is exact
            # level CE
            onehot = np.zeros_like(dist); onehot[np.arange(N), y] = 1.0
            ddist = (dist - onehot) / N
            ce = -np.log(dist[np.arange(N), y] + 1e-12).mean()
            # value MSE against expected level of true label
            yv = np.asarray(q.levels, float)[y]
            dv = 2.0 * (v_raw - yv) / N
            mse = ((v_raw - yv) ** 2).mean()
            loss = ce + mse
            da2 += ddist @ dhead.W
            da2 += dv[:, None] @ vhead.W
            grads[f"h.{q.name}.d.W"] += ddist.T @ a2
            grads[f"h.{q.name}.d.b"] += ddist.sum(0)
            grads[f"h.{q.name}.v.W"] += dv[:, None].T @ a2
            grads[f"h.{q.name}.v.b"] += dv.sum(0)
        else:
            z = c[8]
            p = o["probability"]
            if mode == "ce":
                dz = (p - y) / N  # y is 0/1
                loss = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean()
            else:  # brier
                dz = 2.0 * (p - y) * p * (1.0 - p) / N
                loss = ((p - y) ** 2).mean()
            da2 += dz[:, None] @ model.heads[(q.name, q.kind)].W
            grads[f"h.{q.name}.W"] += dz[:, None].T @ a2
            grads[f"h.{q.name}.b"] += dz.sum(0)

        per_q[q.name] = loss
        total += loss

        # shared question mixer backward (explicit, using per-question cached xq)
        dtanh = (1 - np.tanh(z2) ** 2) * da2
        dxq = dtanh @ model.lq.W
        dlq_W += dtanh.T @ xq
        dlq_b += dtanh.sum(0)
        da1 = dxq[:, : model.d_hidden]
        dQe[_h(q_sig(q), Q_VOCAB)] += dxq[:, model.d_hidden:].sum(0)

        # encoder backward (x is shared across questions)
        dtanh1 = (1 - np.tanh(z1) ** 2) * da1
        dl1_W += dtanh1.T @ x
        dl1_b += dtanh1.sum(0)
        dx = dtanh1 @ model.l1.W
        for (toks, inv), dxi in zip(emb_scatter, dx):
            np.add.at(dE, toks, dxi * inv)

    grads["E"] = dE
    grads["l1.W"], grads["l1.b"] = dl1_W, dl1_b
    grads["lq.W"], grads["lq.b"] = dlq_W, dlq_b
    grads["Qe"] = dQe
    return total, per_q, grads
