"""Calibration metrics and temperature scaling.

A calibrated model has the property Jev/RLCD claims: a 0.80 confidence is
empirically correct ~80% of the time. ECE (expected calibration error) measures
the gap; temperature scaling (one scalar) corrects residual miscalibration.
"""
import numpy as np


def expected_calibration_error(confidences, correct, n_bins=10):
    conf = np.asarray(confidences, float)
    corr = np.asarray(correct, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & ((conf < hi) | (i == n_bins - 1))
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(corr[m].mean() - conf[m].mean())
    return float(ece)


def brier(probs, true_idx):
    """Multi-class Brier (lower = more calibrated). probs: (N, k); true_idx: (N,) int."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(probs.shape[0]), true_idx] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def fit_temperature(Z, targets, init=1.0, lr=0.05, steps=300):
    """Fit scalar T for logits Z/T minimizing NLL. Z: (N,k) untempered logits;
    targets: (N,) int. Returns (T, final_nll)."""
    Z = np.asarray(Z, float)
    N = Z.shape[0]
    onehot = np.zeros_like(Z)
    onehot[np.arange(N), targets] = 1.0
    logT = np.log(float(init))
    for _ in range(steps):
        T = np.exp(logT)
        z = Z / T
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(1, keepdims=True)
        nll = -np.log(p[np.arange(N), targets] + 1e-12).mean()
        dT = -((p - onehot) * Z).sum() / N / (T * T)
        logT -= lr * dT * T
    return float(np.exp(logT)), float(nll)


def reliability(confidences, correct, n_bins=10):
    conf = np.asarray(confidences, float)
    corr = np.asarray(correct, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, cnts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & ((conf < hi) | (i == n_bins - 1))
        cnts.append(int(m.sum()))
        accs.append(float(corr[m].mean()) if m.sum() else float("nan"))
        centers.append((lo + hi) / 2)
    return centers, accs, cnts
