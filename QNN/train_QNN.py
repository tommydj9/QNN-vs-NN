import torch
import torch.nn as nn
import numpy as np

from QNN import QuantumNN
from angle_scaler import load_angle_scaler


def train_qnn(seed, epochs = 200, lr = 0.01):

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_val, X_test, Y_train, Y_val, Y_test, _ = load_angle_scaler()

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(Y_train.values, dtype=torch.float32).unsqueeze(1)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(Y_val.values, dtype=torch.float32).unsqueeze(1)

    # -- MODEL --
    model = QuantumNN()


    # -- loss + optimizer
    loss_fn = nn.BCEWithLogitsLoss() ## Computes loss and convert to logit
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) ## The optimizer decides how to apply the gradients to the nodes based on LR.

    # ----- best-val checkpoint: SAME rule as the NN -----
    best_val_loss = float("inf")


    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad() ## It cleans the gradients at the beggining of every training epoch
        logits = model(Xtr) ## The model makes a guess
        loss = loss_fn(logits, ytr) ## Computes the loss between the actual pred value and the target
        loss.backward() ## Propagates the loss backward, pennylane computes the quantum gradients
        optimizer.step() ## Adam takes the gradient and updates every weight

        # watch Val, never test
        model.eval()
        with torch.no_grad(): ## You don't compute the gradient no more, in eval you're just evaluating the model it doesn have to improve
            val_logits = model(Xva)
            val_loss = loss_fn(val_logits, yva)
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_pred == yva).float().mean()
            pred_rate = val_pred.mean()


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/qnn_AAPL_best.pt")


        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | train_loss {loss.item():.4f} "
                  f"| val_loss {val_loss.item():.4f} | val_acc {val_acc.item():.3f} "
                  f"| pred_rate {pred_rate.item():.2f}")
            
    model.load_state_dict(torch.load("models/qnn_AAPL_best.pt"))
    return model


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_qnn(seed=0)