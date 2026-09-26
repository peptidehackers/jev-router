"""Windowed range-sampler for the saas-sales-conversations CSV.

The 7.17GB CSV stores companies in large contiguous blocks, so a sequential
prefix is company-biased (first 15k rows = 4 companies). Instead: fetch W
disjoint 8MB windows evenly spaced across the file, parse complete rows in
each, dedupe, and keep a company-diverse sample at ~1/10th the download.

Safety: validates on the first window that CSV rows are single-line (no
embedded newlines inside quoted fields); aborts if not, since line-based
window parsing would be invalid.
"""
import csv
import io
import json
import sys
import time
import urllib.request

URL = ("https://huggingface.co/datasets/DeepMostInnovations/"
       "saas-sales-conversations/resolve/main/cleaned_custom_dataset.csv")
TOTAL = 7166710929
WINDOW = 8_000_000
KEEP = ["company_id", "conversation_id", "product_type", "conversation_style",
        "communication_channel", "conversation_flow", "conversation_length",
        "outcome", "customer_engagement", "sales_effectiveness", "full_text"]
MAX_TEXT = 4000


def fetch(off, size):
    req = urllib.request.Request(
        URL, headers={"Range": f"bytes={off}-{off + size - 1}",
                      "User-Agent": "jev-router-loader/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def main(n_windows=80, out_path="real_data.jsonl"):
    hdr = fetch(0, 200000).decode("utf-8", errors="replace")
    header = next(csv.reader(io.StringIO(hdr.split("\n")[0] + "\n")))
    idx = {k: header.index(k) for k in KEEP}
    width = len(header)

    kept = malformed = dupes = truncated = 0
    seen = set()
    companies = set()
    step = (TOTAL - WINDOW) / max(1, n_windows - 1)
    with open(out_path, "w") as out:
        for w in range(n_windows):
            off = int(w * step)
            t0 = time.time()
            text = fetch(off, WINDOW).decode("utf-8", errors="replace")
            lines = text.split("\n")
            body = "\n".join(lines[1:-1])  # drop partial first/last lines
            rows = list(csv.reader(io.StringIO(body)))
            if w == 0 and rows:
                n_lines = len(lines) - 2
                if len(rows) != n_lines:
                    print(f"ABORT: window rows {len(rows)} != lines {n_lines}; "
                          f"quoted newlines present - windowed sampling invalid")
                    sys.exit(2)
            for row in rows:
                if len(row) != width:
                    malformed += 1
                    continue
                try:
                    rec = {k: row[i] for k, i in idx.items()}
                    rec["outcome"] = int(rec["outcome"])
                    rec["customer_engagement"] = float(rec["customer_engagement"])
                    rec["sales_effectiveness"] = float(rec["sales_effectiveness"])
                    rec["conversation_length"] = int(rec["conversation_length"])
                    if not rec["full_text"].strip():
                        malformed += 1
                        continue
                    if len(rec["full_text"]) > MAX_TEXT:
                        rec["full_text"] = rec["full_text"][:MAX_TEXT]
                        truncated += 1
                    cid = rec["conversation_id"]
                    if cid in seen:
                        dupes += 1
                        continue
                    seen.add(cid)
                    companies.add(rec["company_id"])
                    out.write(json.dumps(rec) + "\n")
                    kept += 1
                except (ValueError, KeyError):
                    malformed += 1
            print(f"  window {w + 1}/{n_windows} @ {off/1e9:.2f}GB "
                  f"+{len(rows)} rows ({time.time() - t0:.1f}s) kept={kept}", flush=True)
    print(f"kept={kept} companies={len(companies)} malformed={malformed} "
          f"dupes={dupes} truncated={truncated} -> {out_path}")
    print("companies:", sorted(companies, key=lambda c: int(c.split('-')[1])))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80,
         sys.argv[2] if len(sys.argv) > 2 else "real_data.jsonl")
