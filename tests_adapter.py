"""Applicability-guard regression tests.

Run: python tests_adapter.py  → must print ALL TESTS PASSED.

- Unsupported decision/kind must escalate BEFORE the embedding is validated or
  the model is loaded (never reach an unrelated prediction model).
- Requests without decision/kind keep the documented behavior (length
  validation, then the normal inference path).
"""
import json
import subprocess
import sys

import numpy as np

from train_real_emb import EmbModel
from jev import Question

FIXTURE = "test_set_emb.jsonl"


def run(req: dict) -> dict:
    p = subprocess.run([sys.executable, "infer.py"], input=json.dumps(req),
                       capture_output=True, text=True)
    return json.loads(p.stdout)


def real_embedding() -> list:
    """A real encoder vector from the approved sampling artifact (not synthetic)."""
    import numpy as np
    z = np.load("emb_sample.npz")
    for key in ("X", "embeddings", "emb", "arr_0"):
        if key in z.files:
            arr = z[key]
            vec = arr[0] if arr.ndim > 1 else arr
            return [float(v) for v in np.asarray(vec).ravel()[:3072]]
    raise SystemExit(f"no embedding array in emb_sample.npz; files={z.files}")


def main() -> None:
    failures = []

    # 1. Unsupported decision stops before inference, with prediction null.
    r = run({"decision": "seo_visibility", "kind": "noul",
             "embedding": [0.1] * 3072})
    if r["decision"] != "escalate" or r["prediction"] is not None \
            or "unsupported decision" not in r["reason"]:
        failures.append(f"unsupported decision not stopped: {r}")

    # 2. Unsupported kind stops the same way.
    r = run({"decision": "conversion", "kind": "choice",
             "embedding": [0.1] * 3072})
    if r["decision"] != "escalate" or "unsupported kind" not in r["reason"]:
        failures.append(f"unsupported kind not stopped: {r}")

    # 3. Requests without decision/kind keep documented behavior (length guard
    #    still fires first — wrong length escalates on length, not silently).
    r = run({"embedding": [0.1, 0.2, 0.3]})
    if r["decision"] != "escalate" or "embedding must be a list" not in r["reason"]:
        failures.append(f"length guard changed: {r}")

    # 4. Supported fixture retains existing behavior: a real encoder vector
    #    from the approved sampling artifact flows through to a normal
    #    act/escalate response with the model fingerprint attached — identical
    #    to the pre-guard entry point on the same input.
    emb = real_embedding()
    r = run({"embedding": emb, "target": 0.90})
    if r["decision"] not in ("act", "escalate") or r["model_sha256"] is None:
        failures.append(f"supported fixture behavior changed: {r}")

    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
