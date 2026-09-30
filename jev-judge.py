"""JEV judgment harness: bounded decisions judged from evidence, with an
append-only audit trail.

Doctrine (this is the operating contract):
  - JEV judges ONLY Choice/Score/Noul questions in domains where a head is
    Wilson-certified. Out-of-domain questions fail closed to escalation —
    "calibrated judgment unavailable" — never a guessed probability.
  - Every consequential judgment is recorded: question, domain, evidence,
    result, confidence, tau, threshold version, and the action taken.
  - If verified evidence conflicts with JEV, reality wins (monitor.py drift
    handling already implements this: drift => mandatory fail-closed).

Usage:
  echo '{"domain":"sales-conversion","question":"...","evidence":"...",
         "embedding":[3072 floats],"target":0.90}' | .venv/bin/python jev-judge.py

Audit log: ~/.jev/judgments.jsonl (append-only).
"""
import datetime
import json
import os
import subprocess
import sys

import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(DIR, ".venv", "bin", "python")
INFER = os.path.join(DIR, "infer.py")
AUDIT = os.path.expanduser("~/.jev/judgments.jsonl")

# domains with a certified head. Extend only after approve.py certifies a new
# domain on real labels — never add by hand to make a judgment pass.
CERTIFIED_DOMAINS = {"sales-conversion": {"kind": "noul", "d_in": 3072}}


def record(entry):
    os.makedirs(os.path.dirname(AUDIT), exist_ok=True)
    with open(AUDIT, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    raw = sys.stdin.read() if len(sys.argv) < 2 else open(sys.argv[1]).read()
    try:
        req = json.loads(raw)
        assert isinstance(req, dict)
    except (json.JSONDecodeError, AssertionError):
        out = {"decision": "escalate", "reason": "unparseable judgment request"}
    else:
        domain = req.get("domain", "")
        base = {"question": str(req.get("question", ""))[:500],
                "evidence": str(req.get("evidence", ""))[:2000],
                "domain": domain, "target": req.get("target", 0.90)}
        if domain not in CERTIFIED_DOMAINS:
            out = {"decision": "escalate",
                   "reason": f"domain '{domain}' has no Wilson-certified head; "
                             f"calibrated judgment unavailable — agent/human "
                             f"judgment required. Certified: {sorted(CERTIFIED_DOMAINS)}"}
            record({**base, **out, "action": "deferred-to-agent",
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()})
            print(json.dumps(out))
            return 0
        spec = CERTIFIED_DOMAINS[domain]
        if req.get("kind", spec["kind"]) != spec["kind"]:
            out = {"decision": "escalate",
                   "reason": f"domain '{domain}' certified for kind '{spec['kind']}' only"}
            record({**base, **out, "action": "deferred-to-agent",
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()})
            print(json.dumps(out))
            return 0
        # delegate gating to the certified entry point (single source of truth)
        r = subprocess.run([PY, INFER], input=json.dumps(req).encode(),
                           capture_output=True)
        try:
            out = json.loads(r.stdout)
        except json.JSONDecodeError:
            out = {"decision": "escalate", "reason": f"infer failed: {r.stderr.decode()[:200]}"}
        action = ("auto-acted" if out.get("decision") == "act" else "deferred-to-human")
        record({**base, **{k: out.get(k) for k in
                           ("decision", "prediction", "probability", "confidence",
                            "tau", "threshold_version", "reason")},
                "action": action,
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
