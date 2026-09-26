"""Streaming loader for DeepMostInnovations/saas-sales-conversations.

Downloads the raw CSV once (~71KB/row, dominated by 3072 embedding columns we
do not use), parses incrementally, keeps only the 16 real fields, truncates
extreme conversation lengths, drops malformed/incomplete records (counted),
deduplicates by conversation_id, and writes real_data.jsonl.

Recovered schema (from the dataset card, arXiv 2503.23303):
  outcome                 binary 0/1 conversion          -> Noul primitive
  conversation_style      categorical                     -> Choice primitive
  customer_engagement     0-1 continuous score            -> Score (binned rubric)
  plus leakage-safe split keys: company_id, conversation_id
"""
import csv
import io
import json
import sys
import urllib.request

URL = ("https://huggingface.co/datasets/DeepMostInnovations/"
       "saas-sales-conversations/resolve/main/cleaned_custom_dataset.csv")
KEEP = ["company_id", "conversation_id", "product_type", "conversation_style",
        "communication_channel", "conversation_flow", "conversation_length",
        "outcome", "customer_engagement", "sales_effectiveness", "full_text"]
MAX_TEXT = 4000  # hard cap on input length (documented transformation)


def stream(max_rows: int, out_path: str):
    req = urllib.request.Request(URL, headers={"User-Agent": "jev-router-loader/1.0"})
    n = malformed = dupes = truncated = 0
    seen = set()
    with urllib.request.urlopen(req, timeout=120) as resp, \
            open(out_path, "w") as out, \
            io.TextIOWrapper(resp, encoding="utf-8", errors="replace") as text:
        reader = csv.reader(text)
        try:
            header = next(reader)
        except StopIteration:
            print("ERROR: empty stream", file=sys.stderr)
            return
        idx = {k: header.index(k) for k in KEEP if k in header}
        missing = [k for k in KEEP if k not in header]
        if missing:
            print(f"ERROR: missing columns {missing}", file=sys.stderr)
            return
        for row in reader:
            if n >= max_rows:
                break
            if len(row) != len(header):
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
                out.write(json.dumps(rec) + "\n")
                n += 1
            except (ValueError, KeyError):
                malformed += 1
    print(f"kept={n} malformed_dropped={malformed} duplicates_dropped={dupes} "
          f"truncated_texts={truncated} -> {out_path}")


if __name__ == "__main__":
    stream(int(sys.argv[1]) if len(sys.argv) > 1 else 15000,
           sys.argv[2] if len(sys.argv) > 2 else "real_data.jsonl")
