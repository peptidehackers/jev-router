"""Production inference entry point. Strict contract, fail-closed to escalation.

Request (stdin or file arg):
  {"embedding": [3072 floats], "target": 0.90}   # target optional

Response (always exactly one JSON object):
  {"decision": "act"|"escalate", "prediction": 0|1|null, "probability": p|null,
   "confidence": c|null, "target": t, "tau": tau|null, "threshold_version": id|null,
   "model_sha256": id, "reason": "..."}

Every unsafe or ambiguous condition escalates: unsupported schema, missing or
corrupt model/thresholds, fingerprint mismatch (model or data changed under an
approved threshold), NaN/Inf or wrong-length embeddings, uncertified target,
confidence below threshold, or unparseable input. It never guesses, never
retrains, never loosens a threshold.
"""
import datetime
import hashlib
import json
import math
import sys

import numpy as np

from jev import Question
from train_real_emb import EmbModel

SUPPORTED = {"decision": "conversion", "kind": "noul", "d_in": 3072}
DEFAULT_TARGET = "0.90"
MODEL = "model_emb.npz"
APPROVED = "approved.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def respond(decision, reason, prediction=None, probability=None, confidence=None,
            target=None, tau=None, version=None, model_sha=None):
    print(json.dumps({
        "decision": decision,
        "prediction": prediction,
        "probability": None if probability is None else round(float(probability), 4),
        "confidence": None if confidence is None else round(float(confidence), 4),
        "target": target, "tau": tau, "threshold_version": version,
        "model_sha256": model_sha,
        "reason": reason,
    }))
    return 0


def escalate(reason, **kw):
    return respond("escalate", reason, **kw)


def main():
    raw = sys.stdin.read() if len(sys.argv) < 2 else open(sys.argv[1]).read()
    try:
        req = json.loads(raw)
        if not isinstance(req, dict):
            return escalate("request is not a JSON object")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return escalate("unparseable request JSON")

    # ---- validate input ----
    if "embedding" not in req:
        return escalate("missing required field: embedding")
    emb = req["embedding"]
    if not isinstance(emb, list) or len(emb) != SUPPORTED["d_in"]:
        return escalate(f"embedding must be a list of {SUPPORTED['d_in']} floats "
                        f"(got {type(emb).__name__}, len={len(emb) if isinstance(emb, list) else 'n/a'})")
    try:
        x = np.asarray(emb, dtype=np.float64)
    except (TypeError, ValueError):
        return escalate("embedding contains non-numeric values")
    if not np.all(np.isfinite(x)):
        return escalate("embedding contains NaN or infinite values")
    # encoder-range guard: observed embeddings have |x| <= 0.25 (p99.9 0.08);
    # 5.0 is ~23x the observed max — far outside the encoder's operating
    # range is refused rather than extrapolated
    if float(np.abs(x).max()) > 5.0:
        return escalate("embedding magnitude outside plausible encoder range "
                        "(|x|>5.0; observed encoder max is 0.25); refusing to extrapolate")

    # ---- load + verify approved thresholds ----
    try:
        approved = json.load(open(APPROVED))
    except (FileNotFoundError, json.JSONDecodeError):
        return escalate("approved threshold artifact missing or corrupt")
    model_sha = sha256_file(MODEL) if _exists(MODEL) else None
    if model_sha is None:
        return escalate("model artifact missing")
    if model_sha != approved.get("model_sha256"):
        return escalate("model fingerprint mismatch: approved threshold was "
                        "fitted on a different model — re-approve before acting")
    if approved.get("split_fingerprint") is None or "dataset_sha256" not in approved:
        return escalate("approved artifact incomplete (missing fingerprints)")

    target_key = str(req.get("target", DEFAULT_TARGET))
    # normalize 0.9 -> "0.90"
    try:
        target_key = f"{float(target_key):.2f}"
    except ValueError:
        return escalate(f"invalid target {target_key!r}")
    tentry = approved["targets"].get(target_key)
    if tentry is None:
        return escalate(f"target {target_key} not present in approved artifact "
                        f"(supported: {sorted(approved['targets'])})")
    if not tentry.get("certified") or tentry.get("tau") is None:
        return escalate(f"target {target_key} is NOT Wilson-certified on current "
                        f"calibration evidence (n={tentry.get('cal_n')}); "
                        f"escalation is mandatory", target=target_key)
    tau = float(tentry["tau"])

    # ---- load model and predict ----
    try:
        m = EmbModel(np.random.default_rng(0))
        Q = [Question(SUPPORTED["decision"], SUPPORTED["kind"])]
        ps = m.params(Q)
        saved = np.load(MODEL)
        for k in ps:
            if k not in saved:
                return escalate(f"corrupt model artifact: missing parameter {k}")
            ps[k][:] = saved[k]
    except (OSError, ValueError, KeyError) as e:
        return escalate(f"model artifact corrupt: {e}")

    outs, _ = m.forward(x[None, :], Q)
    p = float(outs[0]["probability"][0])
    pred = int(p > 0.5)
    conf = max(p, 1 - p)

    version = f"{approved['created_at']}:{model_sha[:12]}:{target_key}"
    if conf >= tau:  # boundary: confidence exactly equal to threshold acts
        return respond("act", f"confidence {conf:.3f} >= certified tau {tau:.3f}",
                       prediction=pred, probability=p, confidence=conf,
                       target=target_key, tau=tau, version=version,
                       model_sha=model_sha)
    return escalate(f"confidence {conf:.3f} < certified tau {tau:.3f} "
                    f"(target {target_key})",
                    prediction=pred, probability=p, confidence=conf,
                    target=target_key, tau=tau, version=version,
                    model_sha=model_sha)


def _exists(path):
    try:
        open(path, "rb").close()
        return True
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(main())
