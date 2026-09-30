"""Tests for the JEV judgment harness: domain gating, fail-closed behavior,
and audit-trail integrity.

Run: ./.venv/bin/python tests_judge.py
"""
import json
import os
import subprocess
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(DIR, ".venv", "bin", "python")
AUDIT = os.path.expanduser("~/.jev/judgments.jsonl")
FAILS = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        FAILS.append(name)


def judge(req):
    r = subprocess.run([PY, os.path.join(DIR, "jev-judge.py")],
                       input=json.dumps(req).encode(), capture_output=True, cwd="/")
    return json.loads(r.stdout)


import hashlib
d = __import__("numpy").load(os.path.join(DIR, "emb_sample.npz"))
s = {str(c): int(hashlib.md5(str(c).encode()).hexdigest()[:4], 16) % 100
     for c in d["company_id"]}
GOOD = [float(v) for v in d["X"][[k for k, c in enumerate(d["company_id"])
                                  if s[str(c)] >= "test".__len__() and s[str(c)] >= 85][0]]]

n_before = sum(1 for _ in open(AUDIT)) if os.path.exists(AUDIT) else 0

print("[out-of-domain questions fail closed]")
out = judge({"domain": "code-correctness", "kind": "noul",
             "question": "is this fix complete?", "evidence": "tests pass"})
check("uncertified domain -> escalate, never guess",
      out["decision"] == "escalate" and "no Wilson-certified head" in out["reason"])
out = judge({"domain": "sales-conversion", "kind": "choice",
             "question": "which intent?", "evidence": "x"})
check("wrong kind for certified domain -> escalate",
      out["decision"] == "escalate" and "certified for kind" in out["reason"])

print("[certified domain judges and records]")
out = judge({"domain": "sales-conversion", "kind": "noul",
             "question": "will this conversation convert?",
             "evidence": "held-out conversation",
             "embedding": GOOD, "target": 0.90})
check("certified judgment returns decision + evidence fields",
      out["decision"] in ("act", "escalate") and out.get("threshold_version")
      and out.get("confidence") is not None)

print("[audit trail]")
lines = open(AUDIT).read().strip().split("\n")
entries = [json.loads(l) for l in lines]
check("audit log grew by exactly 3 and is valid JSONL",
      len(entries) == n_before + 3)
check("entries carry question/evidence/result/confidence/action/ts",
      all(k in entries[-1] for k in ("question", "evidence", "decision",
                                     "confidence", "action", "ts"))
      and entries[-1]["action"] in ("auto-acted", "deferred-to-human"))
check("out-of-domain entry recorded with defer action",
      entries[-3]["action"] == "deferred-to-agent"
      and entries[-3]["domain"] == "code-correctness")

print("[malformed input]")
r = subprocess.run([PY, os.path.join(DIR, "jev-judge.py")],
                   input=b"garbage", capture_output=True, cwd="/")
check("unparseable request -> escalate",
      json.loads(r.stdout)["decision"] == "escalate")

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("ALL JUDGE TESTS PASSED")
