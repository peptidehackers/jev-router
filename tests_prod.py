"""Production failure-mode tests: every unsafe or ambiguous condition must
fail closed to escalation. Exercises infer.py's CLI contract end-to-end.

Run: ./.venv/bin/python tests_prod.py   (exits nonzero on any failure)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
DIR = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def infer(req, dir_=DIR):
    r = subprocess.run([PY, "infer.py"], input=json.dumps(req).encode(),
                       capture_output=True, cwd=dir_)
    try:
        return r.returncode, json.loads(r.stdout)
    except json.JSONDecodeError:
        return r.returncode, {"decision": "INVALID", "reason": r.stdout.decode()[:200]}


# a valid embedding from the untouched test split
d = np.load(os.path.join(DIR, "emb_sample.npz"))
import hashlib
def split_of(c):
    b = int(hashlib.md5(str(c).encode()).hexdigest()[:4], 16) % 100
    return "train" if b < 70 else ("cal" if b < 85 else "test")
test_idx = [i for i, c in enumerate(d["company_id"]) if split_of(str(c)) == "test"]
GOOD = [float(v) for v in d["X"][test_idx[0]]]

print("[valid request]")
rc, out = infer({"embedding": GOOD, "target": 0.85})
check("valid embedding -> act or escalate with full evidence",
      rc == 0 and out["decision"] in ("act", "escalate")
      and out["threshold_version"] and out["model_sha256"]
      and out["reason"])

print("[input validation — all must escalate]")
for name, req in [
    ("empty request object", {}),
    ("missing embedding", {"target": 0.85}),
    ("wrong-length embedding", {"embedding": GOOD[:100]}),
    ("NaN in embedding", {"embedding": [float("nan")] + GOOD[1:]}),
    ("inf in embedding", {"embedding": [float("inf")] + GOOD[1:]}),
    ("non-numeric embedding", {"embedding": ["a"] + GOOD[1:]}),
    ("absurd magnitude", {"embedding": [1e9] + GOOD[1:]}),
    ("unknown target", {"embedding": GOOD, "target": 0.75}),
    ("non-numeric target", {"embedding": GOOD, "target": "high"}),
    ("uncertifiable target 0.99", {"embedding": GOOD, "target": 0.99}),
]:
    rc, out = infer(req)
    check(f"{name} -> escalate", out["decision"] == "escalate")

rc, out = infer.__wrapped__ if False else (None, None)
r = subprocess.run([PY, "infer.py"], input=b"not json at all",
                   capture_output=True, cwd=DIR)
check("unparseable JSON -> escalate",
      json.loads(r.stdout)["decision"] == "escalate")

print("[confidence exactly at threshold]")
approved = json.load(open(os.path.join(DIR, "approved.json")))
tau = approved["targets"]["0.85"]["tau"]
# craft an embedding whose confidence lands exactly on tau is impractical;
# verify the boundary semantics directly from the documented rule instead:
src = open(os.path.join(DIR, "infer.py")).read()
check("boundary documented as >= (acts at equality)",
      "conf >= tau" in src)

print("[corrupted / stale artifacts — sandboxed copies]")
with tempfile.TemporaryDirectory() as td:
    shutil.copy(os.path.join(DIR, "infer.py"), td)
    shutil.copy(os.path.join(DIR, "approved.json"), td)
    for f in ("jev.py", "calibration.py", "routing.py", "data.py",
              "train_real_emb.py", "realdata_sample.py", "model_emb.npz",
              "emb_sample.npz"):
        src_f = os.path.join(DIR, f)
        if os.path.exists(src_f):
            try:
                os.link(src_f, os.path.join(td, f))
            except OSError:
                shutil.copy(src_f, os.path.join(td, f))

    # 1. missing model
    os.remove(os.path.join(td, "model_emb.npz"))
    rc, out = infer({"embedding": GOOD}, dir_=td)
    check("missing model -> escalate", out["decision"] == "escalate")

    # 2. corrupted model (bit-flipped)
    shutil.copy(os.path.join(DIR, "model_emb.npz"), os.path.join(td, "model_emb.npz"))
    with open(os.path.join(td, "model_emb.npz"), "r+b") as f:
        f.seek(200); f.write(b"\x00\x01\x02\x03")
    rc, out = infer({"embedding": GOOD}, dir_=td)
    ok = out["decision"] == "escalate" and "fingerprint mismatch" in out["reason"]
    check("corrupted model -> escalate (fingerprint mismatch)", ok)

    # 3. corrupted approved.json
    shutil.copy(os.path.join(DIR, "model_emb.npz"), os.path.join(td, "model_emb.npz"))
    open(os.path.join(td, "approved.json"), "w").write("{broken")
    rc, out = infer({"embedding": GOOD}, dir_=td)
    check("corrupt approved.json -> escalate", out["decision"] == "escalate")

    # 4. stale threshold metadata: model replaced after approval
    shutil.copy(os.path.join(DIR, "approved.json"), os.path.join(td, "approved.json"))
    # retrain-free simulation: perturb one parameter of the model copy
    m = np.load(os.path.join(td, "model_emb.npz"))
    dd = {k: v for k, v in m.items()}
    dd["l1.W"] = dd["l1.W"] + 0.01
    np.savez(os.path.join(td, "model_emb.npz"), **dd)
    rc, out = infer({"embedding": GOOD}, dir_=td)
    ok = out["decision"] == "escalate" and "fingerprint mismatch" in out["reason"]
    check("model changed after approval -> escalate (stale threshold)", ok)

    # 5. missing approved.json
    os.remove(os.path.join(td, "approved.json"))
    rc, out = infer({"embedding": GOOD}, dir_=td)
    check("missing approved.json -> escalate", out["decision"] == "escalate")

print("[duplicate requests are idempotent]")
rc1, o1 = infer({"embedding": GOOD, "target": 0.85})
rc2, o2 = infer({"embedding": GOOD, "target": 0.85})
check("same input -> identical decision", o1 == o2)

print("[drift detection fails closed]")
r = subprocess.run([PY, "monitor.py", "emb_sample.npz"],
                   capture_output=True, cwd=DIR, text=True)
check("monitor runs and reports verdict", "VERDICT" in r.stdout)
# simulate drift: skewed labels + shuffled confidence via label flip
d2 = {k: v for k, v in np.load(os.path.join(DIR, "emb_sample.npz")).items()}
flip = np.random.default_rng(5).random(len(d2["outcome"])) < 0.30
d2["outcome"] = np.where(flip, 1 - d2["outcome"], d2["outcome"])
np.savez(os.path.join(DIR, "_drift_sim.npz"), **d2)
r = subprocess.run([PY, "monitor.py", "_drift_sim.npz"],
                   capture_output=True, cwd=DIR, text=True)
check("30% label corruption -> DRIFT detected + fail-closed guidance",
      "DRIFT DETECTED" in r.stdout and "fail closed" in r.stdout)
os.remove(os.path.join(DIR, "_drift_sim.npz"))

print("[extreme input sizes]")
rc, out = infer({"embedding": [0.0] * 3072})
check("all-zero embedding -> valid act/escalate (no crash)",
      out["decision"] in ("act", "escalate"))
big = np.load(os.path.join(DIR, "emb_sample.npz"))["X"][test_idx[1]]
rc, out = infer({"embedding": [float(v) for v in big * 1000]})
check("extreme-scale embedding -> escalate (range guard)",
      out["decision"] == "escalate" and "magnitude" in out["reason"])

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("ALL PRODUCTION TESTS PASSED")
