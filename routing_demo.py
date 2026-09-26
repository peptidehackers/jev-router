"""End-to-end selective-routing demo: head -> confidence gate -> act / escalate.

Operational question this answers: what fraction of decisions can auto-act at
a guaranteed accuracy level, what escalates to the LLM, and what does the
hybrid cost versus sending everything to the LLM?
"""
import numpy as np

import calibration  # noqa: F401 (kept for extension)
import data
import routing
from demo import QUESTIONS, build_labels, train
from jev import Model

TARGET = 0.93  # most stringent accuracy this calibration split can actually certify


def llm_fallback(rng, n, acc=0.95):
    """LLM stand-in for escalated decisions: correct with prob `acc`,
    independent of the head. Substitute a real model call in production."""
    return rng.random(n) < acc


def main():
    rng = np.random.default_rng(11)
    tr_t, tr_i, tr_u, tr_f = data.make_dataset(3000, seed=1)
    ca_t, ca_i, ca_u, ca_f = data.make_dataset(2000, seed=2)
    ev_t, ev_i, ev_u, ev_f = data.make_dataset(2000, seed=3)
    tr_l = build_labels(tr_i, tr_u, tr_f)
    ca_l = build_labels(ca_i, ca_u, ca_f)
    ev_l = build_labels(ev_i, ev_u, ev_f)

    model = Model(np.random.default_rng(0), d_emb=64, d_hidden=128, d_q=32, d_head=64)
    train(model, tr_t, tr_l, mode="ce", epochs=40, lr=0.1)
    train(model, tr_t, tr_l, mode="brier", epochs=10, lr=0.05)

    # ---- fit thresholds on the calibration split ----
    outs, _, _ = model.forward(ca_t, QUESTIONS)
    by = {o["name"]: o for o in outs}
    sig = {
        "intent": (by["intent"]["confidence"], by["intent"]["choice"] == ca_l["intent"]),
        "urgency": (by["urgency"]["confidence"], by["urgency"]["distribution"].argmax(1) == ca_l["urgency"]),
        "is_fraudulent": (np.maximum(by["is_fraudulent"]["probability"], 1 - by["is_fraudulent"]["probability"]),
                          by["is_fraudulent"]["is_true"].astype(float) == ca_l["is_fraudulent"]),
    }
    thresholds = {name: routing.select_threshold(c, k, TARGET) for name, (c, k) in sig.items()}
    print(f"thresholds selected for {TARGET:.0%} accuracy on auto-acted decisions "
          f"(Wilson 95% lower bound, cal split, n={len(ca_t)}):")
    for name, tau in thresholds.items():
        c, k = sig[name]
        order = np.argsort(-np.asarray(c, float), kind="stable")
        cc, kk = np.asarray(c, float)[order], np.asarray(k, bool)[order]
        accs = np.cumsum(kk) / np.arange(1, len(cc) + 1)
        ok_pt = np.where(accs >= TARGET)[0]
        pt = float(cc[ok_pt[-1]]) if len(ok_pt) else None
        pt_desc = f"{pt:.3f}" if pt is not None else "none"
        if tau is None:
            print(f"  {name:15s} tau=none  -> cannot certify; escalating all "
                  f"(naive point-estimate would have picked tau={pt_desc})")
        else:
            cov, acc = routing.coverage_at(*sig[name], tau)
            print(f"  {name:15s} tau={tau:.3f}  coverage={cov:.1%}  accuracy={acc:.1%}  "
                  f"(naive point-estimate tau={pt_desc})")
    # ---- evaluate the full gate on held-out data ----
    outs, _, _ = model.forward(ev_t, QUESTIONS)
    by = {o["name"]: o for o in outs}
    llm_ok = llm_fallback(rng, len(ev_t))
    print(f"\nheld-out evaluation (n={len(ev_t)}), LLM fallback modeled at 95%:")
    for name, conf, corr in [
        ("intent", by["intent"]["confidence"], by["intent"]["choice"] == ev_l["intent"]),
        ("is_fraudulent", np.maximum(by["is_fraudulent"]["probability"], 1 - by["is_fraudulent"]["probability"]),
         by["is_fraudulent"]["is_true"].astype(float) == ev_l["is_fraudulent"]),
    ]:
        act = routing.Router(thresholds).decide(name, conf)
        cov, acc = act.mean(), corr[act].mean() if act.any() else float("nan")
        sys_acc = ((corr & act) | (llm_ok & ~act)).mean()
        print(f"  {name:15s} auto-act={cov:.1%} @ {acc:.1%}   escalate={1-cov:.1%}   "
              f"head-only={corr.mean():.1%}   system={sys_acc:.1%}")

    # ---- cost model ----
    HEAD_COST, LLM_COST = 0.00001, 0.01  # $ per decision (measured 0.05ms head)
    conf = by["intent"]["confidence"]
    esc = 1 - routing.Router(thresholds).decide("intent", conf).mean()
    routed = len(ev_t) * (HEAD_COST + esc * LLM_COST)
    all_llm = len(ev_t) * LLM_COST
    print(f"\ncost of {len(ev_t)} intent decisions: routed=${routed:.2f} vs all-LLM=${all_llm:.2f} "
          f"({routed / all_llm:.1%})")

    print("\nrisk-coverage (intent, held-out): coverage -> selective accuracy")
    for f, acc in routing.risk_coverage_curve(conf, by["intent"]["choice"] == ev_l["intent"]):
        print(f"   {f:.0%} -> {acc:.1%}")


if __name__ == "__main__":
    main()
