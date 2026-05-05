# kernel_model_jax.py

import pennylane as qml
import jax
import jax.numpy as jnp


class KernelModelJAX:
    """
    JAX-optimized quantum kernel model for FIXED kernels.

    Requirements:
    -------------
    circuit must implement:
        kernel_circuit(x1, x2)

    Notes:
    ------
    - No trainable weights
    - Fully JAX-jittable
    - Supports batched kernel evaluation
    """

    def __init__(
        self,
        circuit,
        device_name="default.qubit",
        interface="jax",
        diff_method="backprop",
        noisy=False,
        shots=1024,
    ):
        if circuit is None:
            raise ValueError("A circuit must be provided.")

        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.interface = interface
        self.diff_method = diff_method
        self.noisy = noisy
        self.circuit_executions = 0

        # Device
        if noisy:
            self.dev = qml.device("default.mixed", wires=self.num_qubits, shots=shots)
        else:
            self.dev = qml.device(device_name, wires=self.num_qubits)

        # QNode
        self._qnode = qml.QNode(
            self.circuit.kernel_circuit,
            self.dev,
            interface=self.interface,
            diff_method=self.diff_method,
        )

        # JIT compile
        self._kernel = jax.jit(self._qnode)

        # Vectorized kernel: K(x1[i], x2[i])
        self._vectorized_kernel = jax.jit(
            jax.vmap(lambda a, b: self._kernel(a, b), in_axes=(0, 0))
        )

    # -------------------------------------------------
    # Pairwise forward (vectorized)
    # -------------------------------------------------
    def forward(self, x1, x2):
        """
        Compute K(x1[i], x2[i]) for all i
        """
        self.circuit_executions += len(x1)
        return self._vectorized_kernel(x1, x2)

    # -------------------------------------------------
    # Full kernel matrix (symmetric)
    # -------------------------------------------------
    def kernel_matrix(self, X):
        """
        Compute full kernel matrix K(X, X)
        """
        def row_fn(x):
            return jax.vmap(lambda x2: self._kernel(x, x2))(X)

        K = jax.vmap(row_fn)(X)

        self.circuit_executions += X.shape[0] ** 2
        return K

    # -------------------------------------------------
    # Rectangular kernel matrix
    # -------------------------------------------------
    def rectangular_kernel_matrix(self, X1, X2):
        """
        Compute K(X1, X2)
        """
        def row_fn(x):
            return jax.vmap(lambda x2: self._kernel(x, x2))(X2)

        K = jax.vmap(row_fn)(X1)

        self.circuit_executions += X1.shape[0] * X2.shape[0]
        return K

    # -------------------------------------------------
    # Efficient pair sampling (for KTA)
    # -------------------------------------------------
    def sampled_kernel(self, X, idx_i, idx_j):
        """
        Compute K(X[i], X[j]) for sampled pairs
        """
        xi = X[idx_i]
        xj = X[idx_j]

        self.circuit_executions += len(idx_i)
        return self._vectorized_kernel(xi, xj)