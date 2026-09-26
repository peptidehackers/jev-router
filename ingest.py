"""One-command real-label ingestion: JSONL -> the npz training contract.

Converts production outcome labels into the exact format train_real_emb.py,
approve.py, and monitor.py consume — no architectural change, no framework.

Input JSONL, one record per line (embedding OR text, not both needed —
embedding preferred; if you use the same 3072-dim encoder as training,
predictions stay in-distribution):

  {"conversation_id": "c-1", "company_id": "acct-9", "outcome": 1,
   "customer_engagement": 0.7, "conversation_style": "warm",
   "embedding": [3072 floats], "full_text": "..."}

Usage:
  python ingest.py labels.jsonl real_labels.npz     # then:
  python train_real_emb.py real_labels.npz && python approve.py && python monitor.py real_labels.npz
"""
import json
import sys

import numpy as np

FIELDS = ("company_id", "conversation_id", "conversation_style",
          "outcome", "customer_engagement", "full_text")


def main(src, dst):
    rows = []
    skipped = 0
    for line in open(src):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            assert "outcome" in r and ("embedding" in r or "full_text" in r)
            r["outcome"] = int(r["outcome"])
            rows.append(r)
        except (json.JSONDecodeError, AssertionError, ValueError, KeyError):
            skipped += 1
    if not rows:
        print("ERROR: no usable records"); return 2
    n_emb = sum(1 for r in rows if "embedding" in r)
    if n_emb != len(rows):
        print(f"NOTE: {len(rows) - n_emb} records lack embeddings; they can be "
              f"monitored (text-only) but not scored by the current head.")
    out = {
        "company_id": np.array([str(r.get("company_id", "unknown")) for r in rows]),
        "conversation_id": np.array([str(r.get("conversation_id", f"row-{i}"))
                                     for i, r in enumerate(rows)]),
        "style": np.array([str(r.get("conversation_style", "unknown")) for r in rows]),
        "outcome": np.array([r["outcome"] for r in rows], dtype=np.int64),
        "engagement": np.array([float(r.get("customer_engagement", 0.0)) for r in rows]),
        "full_text": np.array([str(r.get("full_text", "")) for r in rows]),
    }
    if n_emb == len(rows):
        out["X"] = np.stack([np.asarray(r["embedding"], dtype=np.float32) for r in rows])
    np.savez_compressed(dst, **out)
    print(f"ingested {len(rows)} records ({skipped} skipped) -> {dst}; "
          f"next: train_real_emb.py {dst} && approve.py && monitor.py {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "real_labels.npz"))
