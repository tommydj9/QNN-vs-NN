# Quantum vs Classical Neural Networks for Financial Prediction

A controlled experiment asking one honest question:

> **Does a Quantum Neural Network provide a measurable advantage over a
> classical Neural Network when both are trained on the same financial data?**

The point of this project is **not** to prove quantum computing is better, and
**not** to build a profitable trading strategy. It is to design an experiment
*fair enough to give a disappointing answer* — and then to report that answer
honestly, whatever it turns out to be.

---

## The idea behind the project

Most "AI beats the market" and "quantum beats classical" demos share the same
flaw: the methodology is quietly bent until the desired result appears. Data
gets shuffled so the future leaks into the past. Scalers are fit on the whole
dataset, so the model has secretly seen the test set. Metrics are cherry-picked
— accuracy is reported precisely because it flatters the model. The result looks
impressive and means nothing.

This project is built the opposite way. Every design decision is made to
**remove ways of fooling myself**, so that whatever the models produce can be
trusted:

- **No look-ahead.** Features use only information available *at* the prediction
  date; the target uses the future. Data is split **chronologically** (train →
  validation → test), never shuffled.
- **No leakage.** Scalers are fit on **training data only** and applied to
  validation/test. The model is never allowed to see statistics from the future.
- **A sealed test set.** Test data is opened **exactly once**, at the very end,
  on frozen models — and the result is *not reacted to*. No tuning after the
  reveal.
- **A fair opponent.** The classical baseline is a real neural network with a
  hidden layer, so it can learn feature interactions too. Beating a deliberately
  weak baseline would prove nothing.
- **Honest metrics.** Accuracy is treated as a decoy. The real verdict comes
  from **ROC-AUC**, which cannot be gamed by a model that simply predicts the
  majority class.

The experiment is designed to be **falsifiable**: it can return "QNN better,"
"classical better," "no difference," or "inconclusive," and no outcome is
embarrassing. That is what makes it science rather than salesmanship.

---

## The result

On AAPL weekly data, predicting next-week direction from four technical
features (return, volatility, momentum, volume z-score):

| Model        | Accuracy | ROC-AUC | Brier | pred_rate | Verdict            |
|--------------|----------|---------|-------|-----------|--------------------|
| Classical NN | 49.4%    | 51%   | 25.3% | 25%      | chance             |
| Quantum NN   | 57.7%    | 47.3%   | 24.4% | 97%      | chance (collapsed) |


## Results on multiple seeds


Testing my neural newtworks with different seeds(10) every seed starts with a random set of weights, we calculate the mean and the std dev of the metrics:

| Model        | Mean-Accuracy | std-deviation | Mean Roc-AUC |  ROC_AUC std          | Seeds |
|--------------|----------|---------|-------|-----------|--------------------|
| Classical NN | 52.5%    | +-5.16   | 49% |+-3.61     | 10        |
| Quantum NN   | 47%    | +-6.74   | 48% | +-2.45 | 10 |

**No measurable quantum advantage.** Read carefully, because this table contains
the whole lesson:

Accuracy *appears* to favor the QNN (0.577 vs 0.494) — a naive reading would
declare "quantum wins." It doesn't. The QNN's higher accuracy is an artifact of
**class collapse**: it predicts "up" 97% of the time, and the test period
happened to be up-majority, so accuracy rewards it for guessing the base rate,
not for skill. **ROC-AUC exposes this** — at 0.473 (below 0.5), the QNN cannot
actually distinguish up-weeks from down-weeks any better than a coin flip. Both
models are noise.

> **The core takeaway:** relying on accuracy alone would have produced a
> confident false positive. The guardrails caught it. A trustworthy negative
> result is worth more than an untrustworthy positive one.

A **depth sweep** (1 → 3 entangling layers, 4 → 12 trainable parameters) showed
no improvement, ruling out "the quantum model was starved of capacity" as an
excuse. The binding constraint is the **absence of predictive signal in the
features** — consistent with the efficient-market intuition that a simple
edge like this would already be arbitraged away — not the model class.

**Scope:** this tests one small circuit, one dataset, four near-random features.
It demonstrates no advantage *here* — not that QNNs are useless in general.

---

## How it works

The pipeline is deliberately modular, so the scientifically load-bearing parts
(the data spine) are separate from the models:

```
Raw OHLCV  →  Features (past only)  →  Target (future return)
                        ↓
        Chronological split: train / val / test
                        ↓
        Scale (fit on TRAIN only)
                ↓                    ↓
   StandardScaler              MinMaxScaler [0, π]
        ↓                            ↓
   Classical NN                 Quantum NN (PennyLane)
        ↓                            ↓
        └──── Sealed-test evaluation ────┘
              (opened once, at the end)
```

The two models share **everything** — same features, target, splits, loss
(binary cross-entropy), optimizer (Adam), epochs, and random seed. The *only*
deliberate differences are the model itself and the feature scaling its encoding
requires. That shared spine is what makes any measured difference attributable
to the model rather than to the setup.

**The classical NN** is a small feed-forward network: `4 → 8 (ReLU) → 1 logit`.
The hidden layer + non-linearity is what lets it learn feature interactions.

**The quantum NN** encodes each feature as a rotation angle on its own qubit
(`AngleEmbedding`, which is why features are scaled to `[0, π]`), applies
trainable rotations plus a CNOT entangling layer (`BasicEntanglerLayers` — the
entanglement is what lets the output depend on *joint* feature configurations),
and measures ⟨Z⟩ on one qubit as a logit-like score. It is wrapped as a
`TorchLayer` so it trains with the exact same loop as the classical model.

---

## Project structure

```
├── data.py              # pull & freeze raw OHLCV (single source of truth)
├── features.py          # backward-looking features + future target
├── dataset.py           # join, drop NaNs, chronological split
├── NN/
│   ├── NN.py            # classical neural network
│   └── scaler.py        # StandardScaler, fit on train only
├── QNN/
│   ├── QNN.py           # PennyLane quantum circuit + TorchLayer
│   └── angle_scaler.py  # MinMaxScaler to [0, π] + clip, fit on train only
├── benchmark/
│   └── evaluate.py      # sealed-test evaluation, full metric suite
├── data/                # frozen CSVs + saved scalers
├── models/              # best-val checkpoints
└── NOTES.md             # full results, reasoning, and design decisions
```

---

## Reproducing the experiment

```bash
# 1. fetch and freeze raw data
python data.py

# 2. build features, target, and the chronological split
python dataset.py

# 3. train the classical baseline (saves best-val checkpoint)
python NN/NN.py

# 4. train the quantum model (slow — simulates the circuit per sample)
python QNN/QNN.py

# 5. open the sealed test set ONCE and compare both models
python benchmark/evaluate.py
```

Run from the project root so relative `data/` and `models/` paths resolve.
Random seeds are fixed for reproducibility.

---

## Stack

- **PennyLane** — quantum circuit simulation and differentiation
- **PyTorch** — training loop shared by both models
- **scikit-learn** — scaling and evaluation metrics
- **pandas / yfinance** — data handling and market data

---

## What this project is really about

The financial result is negative. The *methodological* result is the point:
a pipeline where look-ahead bias, data leakage, unfair baselines, and misleading
metrics are each deliberately designed out — so that when the models come back
saying "no advantage," that answer can be believed. Building an experiment that
can prove you *wrong* is harder, and more valuable, than building one that
always agrees with you.
