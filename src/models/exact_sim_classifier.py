import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from ..utils.pauli_utils import generate_pauli_strings, get_pauli_tensor


def pauli_observable(s):
    """Builds the PennyLane observable for a Pauli string ('IXYZ' -> X(1) @ Y(2) @ Z(3))."""
    ops = []
    for idx, char in enumerate(s):
        if char == 'X':
            ops.append(qml.PauliX(idx))
        elif char == 'Y':
            ops.append(qml.PauliY(idx))
        elif char == 'Z':
            ops.append(qml.PauliZ(idx))
    if not ops:
        return qml.Identity(0)          # identity string: expectation is always 1
    prod = ops[0]
    for op in ops[1:]:
        prod = prod @ op
    return prod


def real_pauli_stack(pauli_strings):
    """
    Stacks Pauli matrices as real float64 tensors.

    For real-valued inputs x, x^T P x is real for every Hermitian P, and is
    identically zero when P has an odd number of Y factors (those matrices are
    purely imaginary). Taking .real is therefore exact here, not an
    approximation -- but it must be explicit. The previous implementation passed
    a complex array straight to torch.tensor(dtype=float64), which relies on a
    silent lossy cast and emits a ComplexWarning.
    """
    stack = np.array([get_pauli_tensor(s) for s in pauli_strings])
    return np.ascontiguousarray(stack.real)


def aggregate_sequence(x, b):
    """
    Paper Eq. 5: x_tilde := (1/s) sum_i x_i + b_phi.

    Accepts either (batch, dim) for the non-sequential case s=1, or
    (batch, s, dim) for genuine sequences, which the original implementation
    could not represent.
    """
    if x.dim() == 3:
        x = x.mean(dim=1)
    elif x.dim() != 2:
        raise ValueError(f"expected (batch, dim) or (batch, s, dim), got {tuple(x.shape)}")
    return x + b


class ExactSIMClassifier(nn.Module):
    r"""
    Simplified Hamiltonian (SIM) classifier -- Tiblias et al. (arXiv:2504.10542) Eq. 9.

    .. math::
        f(x) = \sigma\Big( \tfrac{1}{2^n} \sum_j (\tilde x^T P_j \tilde x)\,
                            w_j\, \langle\psi_\theta| P_j |\psi_\theta\rangle \Big)

    Args:
        n_qubits (int): Hilbert space is 2^n_qubits.
        n_layers (int): depth of the strongly-entangling ansatz.
        pauli_strings (list of str): the interaction basis (defaults to all 4^n).
        n_classes (int): 1 for binary (returns probabilities of shape (batch,)).
            For c > 1 the model learns c independent weight vectors w^k sharing a
            single measurement pass, per the one-vs-many scheme in Sec. 3.3, and
            returns logits of shape (batch, c).
        normalize_alpha (bool): apply the 1/2^n prefactor of Eq. 7/9. Defaults to
            False, which reproduces the numbers already in results/. Since w_j is
            free and unconstrained the two settings describe the same hypothesis
            class -- only the initial logit scale differs.
    """

    def __init__(self, n_qubits, n_layers=3, device_name='default.qubit',
                 pauli_strings=None, n_classes=1, normalize_alpha=False):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.dim = 2 ** n_qubits
        self.alpha_scale = 1.0 / self.dim if normalize_alpha else 1.0

        self.pauli_strings = (pauli_strings if pauli_strings is not None
                              else generate_pauli_strings(n_qubits))
        self.n_paulis = len(self.pauli_strings)

        self.dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(self.dev, interface='torch')
        def circuit(weights):
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return [qml.expval(pauli_observable(s)) for s in self.pauli_strings]

        self.qnode = circuit

        self.circuit_weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, 3, dtype=torch.float64))
        # One re-weighting vector per class (Sec. 3.3); shape (1, K) when binary.
        self.w = nn.Parameter(
            (torch.randn(n_classes, self.n_paulis) * 0.01).double())
        self.b = nn.Parameter(torch.zeros(self.dim, dtype=torch.float64))

        self.register_buffer(
            'P_tensor',
            torch.tensor(real_pauli_stack(self.pauli_strings), dtype=torch.float64))

    def pauli_expectations(self):
        """<psi_theta|P_j|psi_theta> -- independent of the input, so evaluated once."""
        e = self.qnode(self.circuit_weights)
        if isinstance(e, (list, tuple)):
            e = torch.stack(e)
        return e

    def classical_features(self, x):
        """alpha_j = (1/2^n) x_tilde^T P_j x_tilde for every string (Eq. 7)."""
        x_tilde = aggregate_sequence(x, self.b)
        feats = torch.einsum('bm,kmn,bn->bk', x_tilde, self.P_tensor, x_tilde)
        return feats * self.alpha_scale

    def forward(self, x):
        feats = self.classical_features(x)              # (B, K)
        e = self.pauli_expectations()                   # (K,)
        logits = feats @ (self.w * e).T                 # (B, c)

        if self.n_classes == 1:
            return torch.sigmoid(logits.squeeze(-1))    # (B,) -- binary, as before
        return logits                                   # (B, c) -- use CrossEntropyLoss
