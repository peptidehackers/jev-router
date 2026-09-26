"""Embedding-carrying sampler: same windowed strategy, keeps the dataset's own
3072-dim embeddings (the SalesRLAgent input representation) as float32, saves
to npz. Fewer windows than the text-only sample (embeddings dominate bytes).
"""
import csv
import io
import sys
import time
import urllib.request

import numpy as np

from realdata_sample import TOTAL, WINDOW, fetch

TEXT_KEEP = ["company_id", "conversation_id", "conversation_style",
             "outcome", "customer_engagement", "full_text"]
MAX_TEXT = 4000
N_EMB = 3072


def main(n_windows=40, out_path="emb_sample.npz"):
    hdr = fetch(0, 200000).decode("utf-8", errors="replace")
    header = next(csv.reader(io.StringIO(hdr.split("\n")[0] + "\n")))
    idx = {k: header.index(k) for k in TEXT_KEEP}
    emb_idx = np.array([header.index(f"embedding_{i}") for i in range(N_EMB)])
    width = len(header)

    recs, X = [], []
    step = (TOTAL - WINDOW) / max(1, n_windows - 1)
    for w in range(n_windows):
        off = int(w * step)
        t0 = time.time()
        text = fetch(off, WINDOW).decode("utf-8", errors="replace")
        lines = text.split("\n")
        rows = list(csv.reader(io.StringIO("\n".join(lines[1:-1]))))
        for row in rows:
            if len(row) != width:
                continue
            try:
                r = {k: row[i] for k, i in idx.items()}
                if not r["full_text"].strip():
                    continue
                X.append(np.asarray([row[i] for i in emb_idx], dtype=np.float32))
                recs.append({
                    "company_id": r["company_id"],
                    "conversation_id": r["conversation_id"],
                    "conversation_style": r["conversation_style"],
                    "outcome": int(r["outcome"]),
                    "customer_engagement": float(r["customer_engagement"]),
                    "full_text": r["full_text"][:MAX_TEXT],
                })
            except (ValueError, KeyError, IndexError):
                continue
        print(f"  window {w + 1}/{n_windows} @ {off/1e9:.2f}GB "
              f"({time.time() - t0:.1f}s) total={len(recs)}", flush=True)

    companies = sorted({r["company_id"] for r in recs})
    np.savez_compressed(out_path,
                        X=np.stack(X),
                        company_id=np.array([r["company_id"] for r in recs]),
                        conversation_id=np.array([r["conversation_id"] for r in recs]),
                        style=np.array([r["conversation_style"] for r in recs]),
                        outcome=np.array([r["outcome"] for r in recs]),
                        engagement=np.array([r["customer_engagement"] for r in recs]),
                        full_text=np.array([r["full_text"] for r in recs]))
    print(f"saved {len(recs)} rows, companies={len(companies)} -> {out_path}")
    print("companies:", companies)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40,
         sys.argv[2] if len(sys.argv) > 2 else "emb_sample.npz")
