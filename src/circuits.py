import pennylane as qml
import jax.numpy as jnp

class ZZFeatureMapKernel:
    def __init__(self, num_qubits, reps=1):
        self.num_qubits = num_qubits
        self.reps = reps
        self.wires = list(range(num_qubits))

    def feature_map(self, x):
        for _ in range(self.reps):
            # Single-qubit encoding
            for i in self.wires:
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)

            # ZZ entanglement
            for i in range(self.num_qubits):
                j = (i + 1) % self.num_qubits
                qml.CNOT(wires=[i, j])
                qml.RZ(x[i] * x[j], wires=j)
                qml.CNOT(wires=[i, j])

    def feature_map_dagger(self, x):
        for _ in reversed(range(self.reps)):
            for i in reversed(range(self.num_qubits)):
                j = (i + 1) % self.num_qubits
                qml.CNOT(wires=[i, j])
                qml.RZ(-x[i] * x[j], wires=j)
                qml.CNOT(wires=[i, j])

            for i in reversed(self.wires):
                qml.RZ(-x[i], wires=i)
                qml.Hadamard(wires=i)

    def kernel_circuit(self, x1, x2):
        self.feature_map(x1)
        self.feature_map_dagger(x2)
        return qml.expval(qml.Projector([0]*self.num_qubits, wires=self.wires))


class AngleReuploadKernel:
    def __init__(self, num_qubits, reps=2, reupload=True):
        self.num_qubits = num_qubits
        self.reps = reps
        self.reupload = reupload
        self.wires = list(range(num_qubits))

    def feature_map(self, x):
        for i in self.wires:
            qml.RY(x[i], wires=i)

    def entanglement(self):
        for i in range(self.num_qubits):
            qml.CNOT(wires=[i, (i+1) % self.num_qubits])

    def layer(self, x):
        self.feature_map(x)
        self.entanglement()

    def feature_map_full(self, x):
        for rep in range(self.reps):
            if rep == 0 or self.reupload:
                self.feature_map(x)
            self.entanglement()

    def feature_map_full_dagger(self, x):
        for rep in reversed(range(self.reps)):
            self.entanglement()  # self-inverse
            if rep == 0 or self.reupload:
                for i in reversed(self.wires):
                    qml.RY(-x[i], wires=i)

    def kernel_circuit(self, x1, x2):
        self.feature_map_full(x1)
        self.feature_map_full_dagger(x2)
        return qml.expval(qml.Projector([0]*self.num_qubits, wires=self.wires))