## Classical NN baseline — results & interpretation

**Setup:** feed-forward NN over 4 technical indicators — momentum,
vol-zscore, volatility, log-return — predicting weekly direction (up/down).

**Result:** validation accuracy landed *below* a coin flip (~0.46), under the
~0.60 base rate of up-weeks. Training loss descended only slowly across 200
epochs and flattened early.

**Interpretation:** four technical indicators carry little to no usable signal
for predicting next-week direction. This is expected, not a bug — if a simple
edge like this existed, it would already be arbitraged away. The fair
experiment gave a fair (disappointing) answer.

**Why this doesn't derail the project:** the research question is *not* "can I
build a profitable strategy." It's:

> Does a QNN perform better than a classical NN **on the same data**?

That's a comparison question. A weak-signal dataset is still a valid — arguably
cleaner — arena for it. What matters is that both models face identical
features, target, and splits, so any measured difference is attributable to the
model, not the data.

**This baseline's role:** it's the reference point the QNN is measured against,
not a number to optimize. Deliberately *not* tuning features/target to lift it,
since tuning against val would leak and corrupt the comparison.

**Open decision:** run the QNN comparison on this weak-signal data as-is (Option
A), or make a bounded, honest attempt to find a target/horizon with more signal
first (Option B). Either is legitimate as long as test stays sealed and the
comparison is reported honestly whatever it shows.


**Today's Question:**
N: 4→8→1 ≈ 49 weights
QNN: (1, 4) = 4 weights

My QNN has ~12× fewer trainable parameters than my classical baseline. 

Capacity isn't just parameter count for quantum circuits. A quantum model's expressivity comes from the structure — entanglement, the exponentially-large state space it implicitly manipulates — not just the raw number of trainable angles. So 4 quantum parameters aren't straightforwardly "less" than 4 classical ones; the comparison is genuinely apples-to-oranges. That's a legitimate point, and it's part of what makes QNN-vs-NN comparison scientifically slippery — you cannot perfectly equalize "capacity" across two such different model families.

**QNN Performance**:
epoch 190 | train_loss 0.6861 | val_loss 0.7363 | val_acc 0.485 | pred_rate 0.88 N_LAYERS = 1


epoch 190 | train_loss 0.6808 | val_loss 0.7206 | val_acc 0.447 | pred_rate 0.88 N_LAYERS = 2

epoch 190 | train_loss 0.6781 | val_loss 0.7116 | val_acc 0.485 | pred_rate 1.00 N_LAYERS = 3

Layers	Params	train_loss	val_acc	pred_rate
1	4	0.686	0.485	0.88
2	8	0.681	0.447	0.88
3	12	0.678	0.485	1.00


Increasing circuit depth from 1 to 3 layers (4 → 12 trainable parameters) did not improve validation accuracy, which remained at chance level. Added capacity instead increased class collapse, with the 3-layer model predicting the majority class for every sample. This indicates the limiting factor is the absence of predictive signal in the features, not model capacity."


**FINAL STATEMENT**:

On identical AAPL weekly data through identical training pipelines, neither the classical NN (test AUC 0.515) nor the QNN (test AUC 0.473) achieved meaningful discrimination — both performed at or near chance. The QNN's higher raw accuracy (0.577 vs 0.494) is an artifact of severe class collapse (predicting "up" 97% of the time) on an up-majority test set, not genuine predictive skill; its below-0.5 AUC confirms no real advantage. No measurable quantum advantage was observed. Combined with the depth sweep (no improvement from 4→12 parameters), this indicates the binding constraint is the absence of predictive signal in the four technical features, not model capacity or model class.


Model	Accuracy	ROC-AUC	Brier	pred_rate	Verdict
NN	0.494	0.515	0.253	0.25	chance
QNN	0.577	0.473	0.244	0.97	chance (collapsed)