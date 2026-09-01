import torch
import torch.nn as nn
import numpy as np
from scaler import load_scaler   # your scaler script


# ----- 1. THE MODEL -----
class ClassicalNN(nn.Module):
    """
    Small feed-forward net for binary direction prediction.
    4 features -> hidden layer (non-linear) -> 1 logit output.
    The hidden layer + ReLU is what lets it learn interactions —
    without it, this collapses to logistic regression.
    """
    def __init__(self, input_size=4, hidden_size=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 4 -> 8 I have 4 features and I turned them in 8 differents neurons creating differents shades of it
            nn.ReLU(),                           # non-linearity: enables curved boundaries, it allows the NN to learn features relationships. If I'd used a logistic regression that'd not have happened
            nn.Linear(hidden_size, 1),           # 8 -> 1 (a single raw score = "logit")
        )
        # NOTE: no sigmoid here on purpose — see the loss choice below. 

    def forward(self, x):
        return self.net(x)   # returns the logit, not the probability. It's the first step


def train_nn(seed, epochs=200, lr=0.01, hidden_size=8):
    torch.manual_seed(seed)   # reproducibility (§14)

    # ----- 2. DATA (scaled, fit on train only — already done in your scaler) -----
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_scaler()

    # to tensors. features float32; targets float32 shaped (N,1) to match the output.
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    model = ClassicalNN(input_size=Xtr.shape[1], hidden_size=hidden_size)

    # ----- 3. LOSS + OPTIMIZER -----
    # BCEWithLogitsLoss = sigmoid + binary cross-entropy fused, numerically stable. Here I use the sigmoid togheter with the loss function, more powerfull if used togheter
    # This is why the model outputs a raw logit and NOT a sigmoid'd probability:
    # the loss applies the sigmoid internally. (The logit/sigmoid thread, paid off.)
    loss_fn = nn.BCEWithLogitsLoss() ## This tells us how much we are wrong
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) ## This moves the weight in that direction buy a small fraction


    best_val_loss = float("inf")

    # ----- 4. THE TRAINING LOOP (the ceremony — identical for the QNN later) -----
    for epoch in range(epochs): ## I repeat the loss and opt pipeline epoch times
        model.train()
        optimizer.zero_grad()        # (a) clear old gradients — they accumulate otherwise
        logits = model(Xtr)          # (b) forward pass: data -> logits
        loss = loss_fn(logits, ytr)  # (c) how wrong are we?
        loss.backward()              # (d) backprop: compute gradients and tells us in which direction change the weight
        optimizer.step()             # (e) nudge weights down the gradient

        # ----- watch VAL, never test -----
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():    # no gradient tracking during evaluation
                val_logits = model(Xva)
                val_loss = loss_fn(val_logits, yva)
                # accuracy: sigmoid -> threshold at 0.5
                val_pred = (torch.sigmoid(val_logits) > 0.5).float()
                val_acc = (val_pred == yva).float().mean()
            print(f"epoch {epoch:3d} | train_loss {loss.item():.4f} "
                  f"| val_loss {val_loss.item():.4f} | val_acc {val_acc.item():.3f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/nn_AAPL_best.pt")

    model.load_state_dict(torch.load("models/nn_AAPL_best.pt"))
    return model ## Return the best model, not the last one in the epoch


if __name__ == "__main__":
    train_nn(seed=0)