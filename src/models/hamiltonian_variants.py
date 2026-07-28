r"""
The two Hamiltonian-classifier variants from Tiblias et al. (arXiv:2504.10542)
that the README advertised but the codebase never contained.

Both evaluate the *full* Hamiltonian expectation of Eq. 2

.. math::  f_{\theta,\phi}(x) = \sigma(\psi_\theta^\dagger H_\phi(x) \psi_\theta)

rather than SIM's truncated sum over p Pauli strings. That is precisely what
separates them from SIM in the paper's complexity table: HAM and PEFF have
sample complexity O(d^2) because they need the whole Hamiltonian, while SIM is
O(p). Evaluating Eq. 2 needs the state vector itself, not a set of Pauli
expectation values, so these models read qml.state() from the circuit.

  HAM  (Sec. 3.1, Eq. 3):  H_phi(x) = H^0_phi + (1/s) sum_i x_i x_i^T
       H^0_phi is a fully parametrized real symmetric matrix -- O(N^2) = O(4^n)
       parameters, which is the cost the paper introduces PEFF to avoid.

  PEFF (Sec. 3.2):         H_phi(x_tilde) = (1/s) sum_i x_tilde_i x_tilde_i^T
       with x_tilde_i := x_i + b_phi, a bias in *input* space -- O(d) parameters.

Note on PEFF: as written, its Hamiltonian is a sum of rank-1 outer products and
is therefore positive semi-definite, so psi^dag H psi >= 0 for every input and
sigma(.) >= 0.5 always -- a fixed 0.5 threshold would emit one class for every
sample. HAM does not have this problem because H^0 is indefinite. We add a
single learnable scalar offset to the logit so PEFF is trainable at all; it
costs one parameter and does not change the model's discriminative content.
This is flagged rather than hidden: the paper's equations omit it, and it is
worth checking against the authors' reference implementation.
"""
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from .exact_sim_classifier import aggregate_sequence


class _StateHamiltonianClassifier(nn.Module):
    """Shared machinery: prepare psi_theta = U_theta|0>, evaluate psi^dag H psi."""

    def __init__(self, n_qubits, n_layers=3, device_name='default.qubit', n_classes=1):
        super().__init__()
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.n_classes = n_classes

        self.dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(self.dev, interface='torch')
        def circuit(weights):
            qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
            return qml.state()

        self.qnode = circuit
        self.circuit_weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, 3, dtype=torch.float64))

    def state(self):
        """psi_theta = U_theta |0>^{otimes n} (Eq. 4), as a complex vector."""
        return self.qnode(self.circuit_weights)

    def _project(self, x_tilde):
        """
        <psi| x_tilde x_tilde^T |psi> = |<psi|x_tilde>|^2 for each sample.

        Computed without ever forming the (dim x dim) outer product, which keeps
        the cost O(2^n) per sample instead of O(4^n).
        """
        psi = self.state()                                  # (dim,) complex
        amp = x_tilde.to(psi.dtype) @ psi.conj()            # (B,) complex
        return (amp.conj() * amp).real                      # (B,) real, >= 0


class HAMClassifier(_StateHamiltonianClassifier):
    """
    Fully-parametrized Hamiltonian (HAM) -- Eq. 2-3.

    H^0_phi is real symmetric (the paper states H^0_phi in R^{NxN}, Hermitian),
    stored as a free matrix M and symmetrized as (M + M^T)/2.
    """

    def __init__(self, n_qubits, n_layers=3, device_name='default.qubit', n_classes=1):
        super().__init__(n_qubits, n_layers, device_name, n_classes)
        # O(N^2) = O(4^n) parameters -- the scaling PEFF exists to reduce.
        self.H0 = nn.Parameter(
            torch.randn(n_classes, self.dim, self.dim, dtype=torch.float64) * 0.01)

    def hamiltonian_bias(self):
        """Symmetrized H^0, guaranteeing a Hermitian (here real symmetric) operator."""
        return 0.5 * (self.H0 + self.H0.transpose(-1, -2))

    def forward(self, x):
        x_tilde = aggregate_sequence(x, torch.zeros(self.dim, dtype=x.dtype,
                                                    device=x.device))
        data_term = self._project(x_tilde)                       # (B,)

        psi = self.state()
        H0 = self.hamiltonian_bias().to(psi.dtype)               # (c, dim, dim)
        bias_term = torch.einsum('i,cij,j->c', psi.conj(), H0, psi).real  # (c,)

        logits = data_term.unsqueeze(-1) + bias_term.unsqueeze(0)  # (B, c)
        if self.n_classes == 1:
            return torch.sigmoid(logits.squeeze(-1))
        return logits


class PEFFClassifier(_StateHamiltonianClassifier):
    """
    Parameter-efficient Hamiltonian (PEFF) -- Sec. 3.2.

    Replaces HAM's O(N^2) matrix bias with an O(d) bias in input space.
    """

    def __init__(self, n_qubits, n_layers=3, device_name='default.qubit', n_classes=1):
        super().__init__(n_qubits, n_layers, device_name, n_classes)
        self.b = nn.Parameter(torch.zeros(self.dim, dtype=torch.float64))
        # See module docstring: without this the PSD Hamiltonian pins sigma(.) >= 0.5.
        self.logit_offset = nn.Parameter(torch.zeros(n_classes, dtype=torch.float64))
        self.logit_scale = nn.Parameter(torch.ones(n_classes, dtype=torch.float64))

    def forward(self, x):
        x_tilde = aggregate_sequence(x, self.b)                  # Eq. 5 bias
        proj = self._project(x_tilde).unsqueeze(-1)              # (B, 1)
        logits = proj * self.logit_scale + self.logit_offset     # (B, c)
        if self.n_classes == 1:
            return torch.sigmoid(logits.squeeze(-1))
        return logits
