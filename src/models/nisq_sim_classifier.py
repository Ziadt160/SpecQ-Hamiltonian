r"""
SIM classifier under a NISQ noise model.

This is `ExactSIMClassifier` with noise channels inserted into the ansatz, so that
noise is the *only* difference between the two. Five defects in the previous
implementation are fixed here; each is noted at the point it occurs.

  1. The ansatz did not match. It hard-coded a nearest-neighbour CNOT ring in
     every layer, whereas StronglyEntanglingLayers cycles the CNOT range as
     [(l mod (n-1)) + 1]. At n=4, L=3 the two circuits differ by up to 0.56 in
     the Pauli expectations *with all noise switched off*, so any
     Exact-vs-NISQ comparison was measuring an ansatz change, not noise.

  2. PhaseDamping was parametrised with T2. Amplitude damping already dephases
     at rate 1/(2 T1), so adding PhaseDamping(1 - exp(-t/T2)) double-counts.
     With the shipped T1=50ms, T2=70ms it realised T2_eff = 58.3 ms rather than
     70 ms. The pure-dephasing time is 1/T_phi = 1/T2 - 1/(2 T1), and the
     channel parameter is 1 - exp(-2t/T_phi).

  3. The defaults were not NISQ-like. gate_time=100ns against T1=50ms makes the
     thermal terms ~2e-6, five thousand times smaller than the two-qubit
     depolarizing rate, so "T1/T2 relaxation" was numerically inert. 50 ms is a
     trapped-ion coherence time; the defaults below are superconducting-scale,
     matching the hardware the README refers to.

  4. Readout error was modelled as a BitFlip channel before measurement. X
     commutes with the flip, so <X> was left *exactly* unchanged while <Y> and
     <Z> were scaled by (1-2p) -- verified numerically. Real readout error is a
     misassignment in whichever basis you measure, so it attenuates every
     non-identity factor equally: <P> -> (1-2p)^weight(P).

  5. The Pauli stack was cast from complex to float64 by a silent lossy
     conversion. `real_pauli_stack` does it explicitly.
"""
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from ..utils.pauli_utils import generate_pauli_strings
from .exact_sim_classifier import (
    aggregate_sequence, pauli_observable, real_pauli_stack)


def entangler_ranges(n_layers, n_qubits):
    """The CNOT ranges StronglyEntanglingLayers uses by default."""
    if n_qubits < 2:
        return [0] * n_layers
    return [(l % (n_qubits - 1)) + 1 for l in range(n_layers)]


def thermal_parameters(t1, t2, gate_time):
    """
    Amplitude- and phase-damping probabilities for one noise insertion point.

    Amplitude damping contributes exp(-t/(2 T1)) to the coherence on its own, so
    the phase channel must supply only the *pure* dephasing that remains:
        1/T_phi = 1/T2 - 1/(2 T1)
    Returns (gamma_amplitude, gamma_phase).
    """
    if t2 > 2 * t1:
        raise ValueError(f"unphysical coherence times: T2={t2} exceeds 2*T1={2*t1}")
    gamma_amp = 1.0 - np.exp(-gate_time / t1)
    inv_t_phi = 1.0 / t2 - 1.0 / (2.0 * t1)
    if inv_t_phi <= 0:
        return gamma_amp, 0.0                     # T1-limited: no pure dephasing
    gamma_phase = 1.0 - np.exp(-2.0 * gate_time * inv_t_phi)
    return gamma_amp, gamma_phase


class NISQSIMClassifier(nn.Module):
    """
    SIM under depolarizing gate noise, T1/T2 relaxation and readout error.

    Defaults are superconducting-scale (T1 = 100 us, T2 = 80 us, 200 ns gates),
    which puts the thermal terms in the same order as the gate depolarizing
    rates instead of five thousand times below them.

    Setting p_gate_1q = p_gate_2q = p_readout = 0 and t1 = t2 = inf reproduces
    `ExactSIMClassifier` exactly.
    """

    def __init__(self, n_qubits, n_layers=3, pauli_strings=None,
                 t1=100e-6, t2=80e-6, gate_time=200e-9,
                 p_gate_1q=0.001, p_gate_2q=0.01,
                 p_readout=0.02,
                 n_classes=1, normalize_alpha=False,
                 device_name='default.mixed'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.dim = 2 ** n_qubits
        self.alpha_scale = 1.0 / self.dim if normalize_alpha else 1.0

        self.pauli_strings = (pauli_strings if pauli_strings is not None
                              else generate_pauli_strings(n_qubits))
        self.n_paulis = len(self.pauli_strings)

        self.p_amp, self.p_phase = thermal_parameters(t1, t2, gate_time)
        self.p_gate_1q = p_gate_1q
        self.p_gate_2q = p_gate_2q
        self.p_readout = p_readout
        self.ranges = entangler_ranges(n_layers, n_qubits)

        # Fix 4: readout misassignment attenuates each measured (non-identity)
        # factor by (1-2p), independent of the basis it is measured in.
        weights = np.array([sum(1 for c in s if c != 'I') for s in self.pauli_strings])
        self.register_buffer(
            'readout_attenuation',
            torch.tensor((1.0 - 2.0 * p_readout) ** weights, dtype=torch.float64))

        self.dev = qml.device(device_name, wires=n_qubits)

        @qml.qnode(self.dev, interface='torch')
        def circuit(w):
            for l in range(n_layers):
                for q in range(n_qubits):
                    qml.Rot(w[l, q, 0], w[l, q, 1], w[l, q, 2], wires=q)
                    self._single_qubit_noise(q)

                # Fix 1: cycle the CNOT range exactly as StronglyEntanglingLayers does.
                if n_qubits > 1:
                    r = self.ranges[l]
                    for q in range(n_qubits):
                        target = (q + r) % n_qubits
                        qml.CNOT(wires=[q, target])
                        for wire in (q, target):
                            if self.p_gate_2q > 0:
                                qml.DepolarizingChannel(self.p_gate_2q, wires=wire)
                            self._thermal(wire)

            return [qml.expval(pauli_observable(s)) for s in self.pauli_strings]

        self.qnode = circuit

        self.circuit_weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, 3, dtype=torch.float64))
        self.w = nn.Parameter((torch.randn(n_classes, self.n_paulis) * 0.01).double())
        self.b = nn.Parameter(torch.zeros(self.dim, dtype=torch.float64))

        self.register_buffer(
            'P_tensor',
            torch.tensor(real_pauli_stack(self.pauli_strings), dtype=torch.float64))

    def _single_qubit_noise(self, wire):
        if self.p_gate_1q > 0:
            qml.DepolarizingChannel(self.p_gate_1q, wires=wire)
        self._thermal(wire)

    def _thermal(self, wire):
        if self.p_amp > 0:
            qml.AmplitudeDamping(self.p_amp, wires=wire)
        if self.p_phase > 0:
            qml.PhaseDamping(self.p_phase, wires=wire)

    def pauli_expectations(self):
        """Noisy <psi|P_j|psi>, including readout attenuation."""
        e = self.qnode(self.circuit_weights)
        if isinstance(e, (list, tuple)):
            e = torch.stack(e)
        return e * self.readout_attenuation

    def classical_features(self, x):
        x_tilde = aggregate_sequence(x, self.b)
        feats = torch.einsum('bm,kmn,bn->bk', x_tilde, self.P_tensor, x_tilde)
        return feats * self.alpha_scale

    def forward(self, x):
        logits = self.classical_features(x) @ (self.w * self.pauli_expectations()).T
        if self.n_classes == 1:
            return torch.sigmoid(logits.squeeze(-1))
        return logits
