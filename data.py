"""Synthetic labeled dataset: support-ticket state -> (intent, urgency, fraud).

The "state" is unstructured text; the decisions are the three primitives. Signal
comes from keyword patterns plus noise, enough to demonstrate that the model
learns calibrated probabilities rather than memorizing.
"""
import numpy as np

INTENTS = ["billing", "technical", "refund", "escalation"]
INTENT_KEYS = {
    "billing": ["invoice", "charge", "payment", "billing", "subscription", "price", "card"],
    "technical": ["crash", "bug", "error", "login", "broken", "slow", "sync", "api"],
    "refund": ["refund", "return", "money back", "cancel", "reimburs"],
    "escalation": ["manager", "supervisor", "legal", "angry", "sue", "executive", "escalat"],
}
URGENCY_KEYS = {
    1: ["just wondering", "whenever", "no rush", "curious", "minor", "cosmetic"],
    2: ["somewhat", "eventually", "annoying", "small", "occasionally"],
    3: ["need", "please", "soon", "today", "problem"],
    4: ["urgent", "asap", "critical", "immediately", "blocking", "deadline"],
    5: ["down", "outage", "emergency", "data loss", "security breach", "on fire", "production down"],
}
FRAUD_KEYS = {
    1: ["gift card", "wire transfer", "western union", "bitcoin", "sensitive info", "social security",
        "urgent payment", "anonymous", "verify account details", "password reset link"],
    0: [],
}


def _ticket(rng):
    intent = rng.integers(0, 4)
    urg = rng.integers(1, 6)
    fraud = 0
    words = []
    for _ in range(rng.integers(3, 8)):
        words.append(rng.choice(INTENT_KEYS[INTENTS[intent]]))
    # urgency signal
    for _ in range(rng.integers(1, 4)):
        words.append(rng.choice(URGENCY_KEYS[int(urg)]))
    # fraud signal, occasionally
    if rng.random() < 0.5:
        fraud = rng.integers(0, 2)
    if fraud == 1:
        for _ in range(rng.integers(2, 5)):
            words.append(rng.choice(FRAUD_KEYS[1]))
    # noise words
    noise = ["customer", "account", "team", "system", "service", "agent", "ticket", "please", "help", "issue"]
    for _ in range(rng.integers(2, 6)):
        words.append(rng.choice(noise))
    rng.shuffle(words)
    text = " ".join(words)
    return text, int(intent), int(urg), fraud


def make_dataset(n, seed=0, ambiguous=0.25, flip=0.05):
    """Uncertainty comes from INPUT ambiguity, not random label flips: a
    fraction `ambiguous` of tickets mix two intents' signal words with the
    label drawn uniformly between them — the Bayes-optimal prediction on those
    is genuinely ~50/50, so a calibrated model must be uncertain. A small
    `flip` adds residual label noise. (Pure label noise makes confidence
    separation — and therefore selective routing — impossible by
    construction: a flipped label looks identical to the input.)"""
    rng = np.random.default_rng(seed)
    texts, intents, urgs, frauds = [], [], [], []
    for _ in range(n):
        t, i, u, f = _ticket(rng)
        if rng.random() < ambiguous:
            j = int(rng.integers(0, 4))
            while j == i:
                j = int(rng.integers(0, 4))
            extra = [str(rng.choice(INTENT_KEYS[INTENTS[j]]))
                     for _ in range(int(rng.integers(3, 7)))]
            t = t + " " + " ".join(extra)
            i = i if rng.random() < 0.5 else j
        if rng.random() < flip:
            i = int(rng.integers(0, 4))
        if rng.random() < flip:
            u = int(rng.integers(1, 6))
        if rng.random() < flip:
            f = 1 - f
        texts.append(t); intents.append(i); urgs.append(u); frauds.append(f)
    return texts, np.asarray(intents), np.asarray(urgs), np.asarray(frauds)


def labels(intents, urgs, frauds):
    return {"intent": intents, "urgency": urgs, "is_fraudulent": frauds.astype(float)}
