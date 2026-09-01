import pennylane as qml
import torch
import torch.nn as nn 
import numpy as np
from angle_scaler import load_angle_scaler


N_QUBITS = 4 # 1 per feature - WIDTH
N_LAYERS = 3 # Entrangling layers - DEPTH

dev = qml.device("default.qubit", wires = N_QUBITS)


@qml.qnode(dev, interface = "torch")
def circuit(inputs, weights):

    # inputs: 4 features for ONE sample, scaled into [0, pi]
    # wights: trainable rotation angles, shape(N_LAYERS, N_QUBITS)

    qml.AngleEmbedding(inputs, wires=range(N_QUBITS)) ## Every feature now becomes an RX rotation on his own qubit

    # Entanglment Layers allows the output to depend on JOINT configurations of features finding better patterns.
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))


    return qml.expval(qml.PauliZ(0)) ## Returns the exp_val of qubit 0 in range [-1: 1]

class QuantumNN(nn.Module):

    """Wraps the QNode as a torch layer -> trains with the SAME loop as the NN."""

    def __init__(self):
        super().__init__()
        weights_shape = {"weights": (N_LAYERS, N_QUBITS)} # You have a rotation per qubit per layer, if you have 3 layers of 4 qubits you'll have 12 trainable angles
        self.qlayer = qml.qnn.TorchLayer(circuit, weights_shape) ## Masking this circuit as a pytorch objet so we can inherit the built-in method of pytorch

    def forward(self, x):
        return self.qlayer(x).unsqueeze(1)

    """
        When you write self.qlayer(x), here's the sequence it runs, invisibly:

        takes your input tensor x (a batch of angle-scaled features),
        feeds each sample's features into the circuit as the inputs,
        supplies its own stored weights as the circuit's weights,
        runs the quantum simulation,
        returns the ⟨Z⟩ measurement per sample, as a PyTorch tensor.

    """

if __name__ == "__main__":
    # feed it REAL angle-scaled data — no training yet, just prove it runs
    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, _ = load_angle_scaler()

    model = QuantumNN()
    x = torch.tensor(X_train_s[:5], dtype=torch.float32)   # first 5 samples
    out = model(x)

    print("input shape :", x.shape)
    print("output shape:", out.shape)
    print("outputs (<Z> per sample):\n", out.detach().numpy().round(4))