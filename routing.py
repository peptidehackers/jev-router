"""Selective prediction: route on calibrated confidence.

The anti-hallucination mechanism is not the head's accuracy — it's the gate:
decisions above a confidence threshold auto-act (no LLM, no hallucination
possible), decisions below escalate to an LLM or human. Thresholds are chosen
on a held-out calibration split to meet a required accuracy on auto-acted
decisions while maximizing coverage.
"""
import numpy as np


def _wilson_lb(k, n, z=1.96):
    """Wilson score interval lower bound for k successes in n trials."""
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - margin) / denom


def select_threshold(confidences, correct, target_accuracy):
    """Largest-coverage threshold whose *Wilson lower bound* (95%) on prefix
    accuracy meets `target_accuracy`. Conservative by construction: on a
    finite calibration split the point estimate can be optimistically noisy;
    the lower bound cannot. Returns None if nothing qualifies — then every
    decision escalates.
    """
    conf = np.asarray(confidences, float)
    corr = np.asarray(correct, bool)
    order = np.argsort(-conf, kind="stable")
    c, k = conf[order], corr[order]
    n = len(c)
    cum = np.cumsum(k)
    sizes = np.arange(1, n + 1)
    lbs = np.array([_wilson_lb(int(cum[s - 1]), int(s)) for s in sizes])
    ok = np.where(lbs >= target_accuracy)[0]
    if len(ok) == 0:
        return None
    s = ok[-1] + 1  # largest prefix whose lower bound meets the target
    return float(c[s - 1])


def coverage_at(confidences, correct, tau):
    conf = np.asarray(confidences, float)
    corr = np.asarray(correct, bool)
    m = conf >= tau
    if m.sum() == 0:
        return 0.0, float("nan")
    return float(m.mean()), float(corr[m].mean())


def risk_coverage_curve(confidences, correct, n_points=10):
    conf = np.asarray(confidences, float)
    corr = np.asarray(correct, bool)
    order = np.argsort(-conf, kind="stable")
    n = len(conf)
    return [(f, float(corr[order[:max(1, int(round(f * n)))]].mean()))
            for f in np.linspace(0.1, 1.0, n_points)]


class Router:
    """Per-question thresholds. decide(name, conf) -> bool array: True = act."""

    def __init__(self, thresholds):
        self.thresholds = thresholds  # name -> tau (None = always escalate)

    def decide(self, name, conf):
        conf = np.asarray(conf, float)
        tau = self.thresholds.get(name)
        if tau is None:
            return np.zeros(len(conf), bool)
        return conf >= tau
