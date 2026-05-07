import pennylane as qml
import jax.numpy as jnp
import jax
import numpy as np

class ZZFeatureMapKernel:
    def __init__(self, num_qubits, reps=1):
        self.num_qubits = num_qubits
        self.reps = num_qubits - 1
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
        self.reps = num_qubits - 1
        self.reupload = True
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

class quackEmbeddingCircuit:
    """
        JAX-safe angle-embedding quantum kernel.
        All inputs MUST already have shape (num_qubits,).
        No shape logic is allowed inside the circuit.

        Args:
            num_qubits  : number of qubits
            reps        : number of circuit repetitions
            reupload    : whether to re-upload features each rep
            noisy       : if True, apply depolarising noise after every gate
                        on BOTH the forward (x1) and manual-adjoint (x2) passes
            noise_level : depolarising error probability p ∈ [0, 1]

        Device requirement
        ------------------
        noisy=False  →  default.qubit  (state-vector)
        noisy=True   →  default.mixed  (density-matrix)
    """
    def __init__(self, num_qubits, reps=1, reupload=True, noisy=False, noise_level=0.01):
        self.num_qubits  = num_qubits
        self.reps        = reps
        self.reupload    = reupload
        self.noisy       = noisy
        self.noise_level = noise_level
        self.wires       = list(range(num_qubits))

    def _depolarise(self, wire, apply_noise):
        if self.noisy and apply_noise:
            qml.DepolarizingChannel(self.noise_level, wires=wire)

    def feature_map(self, x, scale, apply_noise=True):
        scale = qml.math.asarray(scale)   # ← backend-agnostic
        for i, wire in enumerate(self.wires):
            qml.Hadamard(wires=wire)
            self._depolarise(wire, apply_noise)
            qml.RZ(scale[i] * x[i], wires=wire)
            self._depolarise(wire, apply_noise)

    def _feature_map_dagger(self, x, scale, apply_noise=True):
        scale = qml.math.asarray(scale)
        for i in reversed(range(self.num_qubits)):
            wire = self.wires[i]
            qml.RZ(-scale[i] * x[i], wires=wire)
            self._depolarise(wire, apply_noise)
            qml.Hadamard(wires=wire)
            self._depolarise(wire, apply_noise)

    def ansatz(self, var, rot, apply_noise=True):
        var = qml.math.asarray(var)
        rot = qml.math.asarray(rot)
        for i, wire in enumerate(self.wires):
            qml.RY(var[i], wires=wire)
            self._depolarise(wire, apply_noise)
        for i in range(self.num_qubits):
            target = self.wires[(i + 1) % self.num_qubits]
            qml.CRZ(rot[i], wires=[self.wires[i], target])
            self._depolarise(self.wires[i], apply_noise)
            self._depolarise(target, apply_noise)

    def _ansatz_dagger(self, var, rot, apply_noise=True):
        var = qml.math.asarray(var)
        rot = qml.math.asarray(rot)
        for i in reversed(range(self.num_qubits)):
            target = self.wires[(i + 1) % self.num_qubits]
            qml.CRZ(-rot[i], wires=[self.wires[i], target])
            self._depolarise(self.wires[i], apply_noise)
            self._depolarise(target, apply_noise)
        for i in reversed(range(self.num_qubits)):
            wire = self.wires[i]
            qml.RY(-var[i], wires=wire)
            self._depolarise(wire, apply_noise)

    def _build_circuit(self, x, weights, apply_noise=True):
        idx = 0
        for rep in range(self.reps):
            scale = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            var   = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            rot   = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            if rep == 0 or self.reupload:
                self.feature_map(x, scale, apply_noise)
            self.ansatz(var, rot, apply_noise)

    def _build_circuit_dagger(self, x, weights, apply_noise=True):
        params, idx = [], 0
        for _ in range(self.reps):
            scale = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            var   = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            rot   = weights[idx: idx + self.num_qubits]; idx += self.num_qubits
            params.append((scale, var, rot))
        for rep in reversed(range(self.reps)):
            scale, var, rot = params[rep]
            self._ansatz_dagger(var, rot, apply_noise)
            if rep == 0 or self.reupload:
                self._feature_map_dagger(x, scale, apply_noise)

    def kernel_circuit(self, x1, x2, weights):
        self._build_circuit(x1, weights, apply_noise=True)
        self._build_circuit_dagger(x2, weights, apply_noise=True)
        return qml.expval(qml.Projector([0] * self.num_qubits, wires=self.wires))

    def init_weights(self, seed=0, minval=-jnp.pi, maxval=jnp.pi):
        rng = np.random.default_rng(seed)
        total = self.reps * 3 * self.num_qubits
        return rng.uniform(minval, maxval, size=(total,)).astype(jnp.float64)

    def init_weights_jax(self, seed=0, minval=-jnp.pi, maxval=jnp.pi):
        """Use this variant only in sim mode."""
        key = jax.random.PRNGKey(seed)
        total = self.reps * 3 * self.num_qubits
        return jax.random.uniform(key, shape=(total,), minval=minval, maxval=maxval)
