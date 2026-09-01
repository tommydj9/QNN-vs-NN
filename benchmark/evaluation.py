import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent          # benchmark/ -> quantum_computing/
sys.path.append(str(ROOT / "NN"))
sys.path.append(str(ROOT / "QNN"))

print("ROOT:", ROOT)
print((ROOT / "NN" / "scaler.py").exists())
print((ROOT / "QNN" / "angle_scaler.py").exists())

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    f1_score, confusion_matrix, log_loss,
)

from scaler import load_scaler            # NN/scaler.py  (NOT NN.py)
from angle_scaler import load_angle_scaler   # QNN/angle_scaler.py  (NOT QNN.py)
from NN import ClassicalNN, train_nn                    # NN/NN.py
from QNN import QuantumNN                      # QNN/QNN.py
from train_QNN import train_qnn



def evaluate_model(model, X_test, y_test, name):
    """Run one frozen model on the SEALED test set. Called ONCE, at the end."""
    model.eval()
    with torch.no_grad():
        Xte = torch.tensor(X_test, dtype=torch.float32)
        logits = model(Xte)
        probs = torch.sigmoid(logits).numpy().ravel()   # <- probability of "up"

    preds = (probs > 0.5).astype(int)
    y = y_test.values

    metrics = {
        "accuracy": accuracy_score(y, preds),        # the DECOY — inflated by base rate
        "roc_auc":  roc_auc_score(y, probs),         # the TRUTH — 0.5 = no discrimination
        "brier":    brier_score_loss(y, probs),      # calibration, lower better
        "log_loss": log_loss(y, probs),
        "f1":       f1_score(y, preds),
        "pred_rate": preds.mean(),                   # collapse check: ~1.0 = always "up"
    }

    print(f"\n=== {name} (SEALED TEST) ===")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print("  confusion matrix [ [TN FP] [FN TP] ]:")
    print("   ", confusion_matrix(y, preds).tolist())
    return metrics


def nn_shuffle_seeds(seeds, feats, target):
    results = []
    roc_auc = []
    for s in seeds:
        model = train_nn(s)
        m = evaluate_model(model, feats, target, f"NN seed {s}")
        results.append(m["accuracy"])
        roc_auc.append(m["roc_auc"])
    return np.mean(results), np.std(results), np.mean(roc_auc), np.std(roc_auc)

def qnn_shuffle_seeds(seeds, feats, target):
    results = []
    roc_auc = []
    for s in seeds:
        model = train_qnn(s)
        m = evaluate_model(model, feats, target, f"QNN seed {s}")
        results.append(m["accuracy"])
        roc_auc.append(m["roc_auc"])
    return np.mean(results), np.std(results), np.mean(roc_auc), np.std(roc_auc)

    

    

if __name__ == "__main__":
    # ---- NN on its standard-scaled test set ----
    Xtr_s, Xva_s, Xte_s, ytr, yva, yte, _ = load_scaler()
    nn_model = ClassicalNN(input_size=4, hidden_size=8)
    nn_model.load_state_dict(torch.load(ROOT / "models" / "nn_AAPL_best.pt"))
    evaluate_model(nn_model, Xte_s, yte, "Classical NN")


    print(f"\n Mean Metrics of NN model based on different seeds")
    nn_mean, nn_std, nn_roc, nn_roc_std = nn_shuffle_seeds(list(range(10)), Xte_s, yte)
    print(f"MEAN ACC: {nn_mean*100:.2f}%, STD: +-{nn_std*100:.2f}, MEAN ROC-AUC: {nn_roc*100:.2f}%, STD: +-{nn_roc_std*100:.2f}")


    # ---- QNN on its angle-scaled test set ----
    Xtr_a, Xva_a, Xte_a, ytr_a, yva_a, yte_a, _ = load_angle_scaler()
    qnn_model = QuantumNN()
    qnn_model.load_state_dict(torch.load(ROOT / "models" / "qnn_AAPL_best.pt"))
    evaluate_model(qnn_model, Xte_a, yte_a, "Quantum NN")

    print(f"\n Mean Metrics of QNN model based on different seeds")
    qnn_mean, qnn_std, qnn_roc, qnn_roc_std = qnn_shuffle_seeds(list(range(10)), Xte_a, yte_a)
    print(f"MEAN ACC: {qnn_mean*100:.2f}%, STD: +-{qnn_std*100:.2f}MEAN ROC-AUC: {qnn_roc*100:.2f}%, STD: +-{qnn_roc_std*100:.2f}")

