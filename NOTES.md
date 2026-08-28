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