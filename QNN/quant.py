import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)  #We create a quantum device using the default.qubit simulator with 2 qubits. This allows us to simulate the quantum circuit on a classical computer.

class QuantumNeuralNetwork:
    

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size, requires_grad=True)  #We initialize the weights of the quantum neural network randomly. The weights are a 2-dimensional vector that will be updated during training.
        self.TRAINING_STEPS = 100  #We set the number of training steps to 100. This means that the training loop will run for 100 iterations.

    ## create a quantum circuit with 2 qubits, actually we are recreating the mathematical enviroment of the quantum circuit in a classical computer, 
    # so we can simulate it and get the results of the circuit without having to use a real quantum computer.
    # Quantum nreutal network with 2 qubits and 1 layer

    @qml.qnode(dev) ## This tells the compiler that this function is a quantum node, which means that it can be executed on a quantum device. The qnode decorator takes the quantum device as an argument, which specifies the device on which the circuit will be executed.
    def circuit(self, x, weights):

        # 1) Encoding the input data into the quantum circuit, Represent my classical data as a quantum state.

        qml.RX(x[0], wires=0) # This encodes the normal data into quantum system data using rotation gates. The input data is a 2-dimensional vector x, where x[0] is encoded using an RX gate on qubit 0 and x[1] is encoded using an RY gate on qubit 1.
        qml.RY(x[1], wires=1)  #First we encode the input data into the quantum circuit using rotation gates. The input data is a 2-dimensional vector x, where x[0] is encoded using an RX gate on qubit 0 and x[1] is encoded using an RY gate on qubit 1.

        qml.CNOT(wires=[0, 1])  #Then we apply a CNOT gate to entangle the two qubits. The control qubit is qubit 0 and the target qubit is qubit 1.

        qml.RY(weights[0], wires=0)  #Next, we apply a rotation gate to qubit 0 using the first weight parameter. This allows the circuit to learn from the input data.
        qml.RY(weights[1], wires=1)  #Then we apply a rotation gate to qubit 1 using the second weight parameter. This allows the circuit to learn from the input data.
 
        return (qml.expval(qml.PauliZ(0)) + 1) / 2  #Finally, we measure the expectation value of the Pauli-Z operator on qubit 0. We standardize the output to be in the range [0, 1] by adding 1 and dividing by 2. This allows us to compare the output of the circuit with the target value during training.

        # You take the value in the quantum system and tranform it back to classical numeration using PAULI_Z
        # the with expval you check the expected value of that pauli_z operator in the long run.

        ##This value states between +-1, so either we standirdize it into a 0-1 range or we use it as it is, depending on the problem we are trying to solve.


    def train(self, x, y):
        self.opt = qml.GradientDescentOptimizer(stepsize=0.1)  #We create an optimizer using the Gradient Descent algorithm with a step size of 0.1. This will be used to update the weights of the circuit during training.

        for i in range(self.TRAINING_STEPS): #We run the training loop for a specified number of steps. In each step, we compute the gradient of the loss function with respect to the weights and update the weights using the optimizer.

            self.weights = self.opt.step(
                lambda w: (self.circuit(x, w) - y) ** 2
            )

    


## I cannot entangle 4 qubits togheter just using 1 CNOT gate, CNOT is a 2 qubit gate, 
# so I need to use more CNOT gates to entangle all the qubits together. 
# For example, I can use 3 CNOT gates to entangle 4 qubits together. 
# The first CNOT gate will entangle qubit 0 and qubit 1, the second CNOT gate will entangle qubit 1 and qubit 2,
# and the third CNOT gate will entangle qubit 2 and qubit 3. This way, all the qubits will be entangled together.
# THAT"S CALLED ENTTANGLING LAYER, and it is a common technique used in quantum circuits to create entanglement between multiple qubits.


# But the more your circuit-depth grows the more noise you will have in your circuit, the more difficult the optimization will be, 
# and the more difficult it will be to train your quantum neural network.
# Landscape flattens out almost everywhere, 
# so your optimizer gets no signal about which way to move the weights and training stalls, 
# that's an open problem in quantum machine learning, and it is called barren plateaus.


# Another constraint: for every qubit you add to your circuit(feature), the number of parameters you need to optimize grows exponentially,
# every time the optimizer has to walk trough the whole vector. Let's suppose you have N qubits:
# every single step inside the training the optimizer has to walk trough a vector of size 2^N, 
# and that is a problem because the more qubits you add the more parameters you have to optimize, 
# and the more difficult it will be to train your quantum neural network. That's the real wall of quantum machine learning, and it is called the curse of dimensionality.


## Pauli_Z operator is a quantum operator that tells you "is this qubit 0 or 1?" 
# but reports the answer as +1 or −1 rather than as 0 or 1
# it simply does the exp_value = 1 * (probability of measuring 0) -1 * (probability of measuring 1)
# you'll get a value between -1 and 1, like 0.08, but my target lives in [0 ,1], so I need to standardize it to be in the same range as my target, 
# and I can do that by using the formula: (exp_value + 1) / 2