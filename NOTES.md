# QNN vs Classical NN — AAPL Weekly Direction

**Research question:** Does a QNN perform better than a classical NN
**on the same data**? (A comparison question — *not* "build a profitable
strategy.")

---

## Final result (sealed test)

Neither model achieved meaningful discrimination — both at/near chance.

| Model | Accuracy | ROC-AUC | Brier | pred_rate | Verdict |
|-------|----------|---------|-------|-----------|---------|
| NN  | 0.494 | 0.515 | 0.253 | 0.25 | chance |
| QNN | 0.577 | 0.473 | 0.244 | 0.97 | chance (collapsed) |

**No measurable quantum advantage.** The QNN's higher accuracy (0.577 vs
0.494) is a **class-collapse artifact** — it predicts "up" 97% of the time
on an up-majority test set, so accuracy rewards it for the base rate, not
skill. Its **AUC of 0.473 (below 0.5)** confirms no real discrimination.

> **Key lesson:** accuracy alone would have produced a false positive
> ("QNN wins, 58% > 49%"). AUC exposed that both models are noise. This is
> exactly why accuracy is not trusted alone on imbalanced data.

**Scope caveat:** this tests *one* small circuit, *one* dataset, *four*
near-random features. It shows no advantage *here* — not that QNNs are
useless in general.

---

## Why both models are at chance: the data, not the model

Four technical indicators (momentum, vol-zscore, volatility, log-return)
carry little usable signal for next-week direction. Expected, not a bug —
a simple edge like this would already be arbitraged away.

**Evidence it's signal, not capacity — the depth sweep:**

| Layers | Params | train_loss | val_acc | pred_rate |
|--------|--------|-----------|---------|-----------|
| 1 | 4  | 0.686 | 0.485 | 0.88 |
| 2 | 8  | 0.681 | 0.447 | 0.88 |
| 3 | 12 | 0.678 | 0.485 | 1.00 |

Tripling parameters (4→12) did not improve val accuracy — it stayed at
chance, and class collapse got *worse* (pred_rate → 1.00 at 3 layers).
Note train_loss dropped with depth, but this reflects the model committing
harder to the majority class, **not** learning. This rules out "the QNN was
starved of capacity" as the explanation.

---

## The parameter-count asymmetry (and why it's not the story)

NN: 4→8→1 ≈ 49 weights. QNN: (1,4) = 4 weights. ~12× fewer.

But capacity isn't just parameter count for quantum circuits — expressivity
comes from structure (entanglement, the exponentially-large state space),
not raw angle count. So 4 quantum params aren't straightforwardly "less"
than 49 classical ones; the comparison is genuinely apples-to-oranges, and
you can't perfectly equalize capacity across two such different model
families. The depth sweep above is what lets us set this objection aside:
more quantum params didn't help.

---

## Method discipline (why the result is trustworthy)

- Identical features, target, splits, loss, optimizer, epochs, seed across
  both models — only model + required scaling differ.
- Baseline deliberately **not** tuned to look better (tuning against val
  would leak).
- Test opened **once**, at the end, on frozen models. Not reacted to.

## Open threads

- Confirm QNN checkpoint was saved under the corrected `<` rule (best val),
  not the earlier backwards `>`.
- Possible extension: a target/horizon with more signal (bounded, honest),
  or a backtest layer (§12).