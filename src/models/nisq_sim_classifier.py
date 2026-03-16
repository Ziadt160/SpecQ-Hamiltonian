import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from ..utils.pauli_utils import generate_pauli_strings, get_pauli_tensor

class NISQSIMClassifier(nn.Module):
    """
    NISQ SIM Classifier implementing Eq 9 from the paper under realistic noise models.
    
    Noise Models implemented (on default.mixed):
    1. Thermal Relaxation (T1, T2): Modeled via AmplitudeDamping and PhaseDamping.
    2. Gate Noise: Modeled via DepolarizingChannel.
    3. Readout Error: Modeled via BitFlip before measurement.
    """
    def __init__(self, n_qubits, n_layers=3, pauli_strings=None, 
                 t1=50e-3, t2=70e-3, gate_time=100e-9,  # Thermal params (seconds)
                 p_gate_1q=0.001, p_gate_2q=0.01,       # Gate depolarizing probs
                 p_readout=0.02,                        # Readout flip prob
                 device_name='default.mixed'):
        super().__init__()
        self.n_qubits = n_qubits
        if pauli_strings is not None:
            self.pauli_strings = pauli_strings
        else:
            self.pauli_strings = generate_pauli_strings(n_qubits)
        self.n_paulis = len(self.pauli_strings)
        
        # Noise Parameters calculation
        # Damping probability = 1 - exp(-t / T)
        self.p_amp = 1 - np.exp(-gate_time / t1)
        self.p_phase = 1 - np.exp(-gate_time / t2)
        
        self.p_gate_1q = p_gate_1q
        self.p_gate_2q = p_gate_2q
        self.p_readout = p_readout

        # 1. Quantum Device & Circuit (Mixed State)
        self.dev = qml.device(device_name, wires=n_qubits)
        
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs):
            # inputs is (n_layers, n_qubits, 3) for StronglyEntangling
            # We manually implement layers to inject noise
            
            for l in range(n_layers):
                # 1. Rotations + Noise
                for q in range(n_qubits):
                    qml.Rot(inputs[l, q, 0], inputs[l, q, 1], inputs[l, q, 2], wires=q)
                    # Single Qubit Noise
                    qml.DepolarizingChannel(self.p_gate_1q, wires=q)
                    qml.AmplitudeDamping(self.p_amp, wires=q)
                    qml.PhaseDamping(self.p_phase, wires=q)

                # 2. Entanglement (CNOT chain) + Noise
                for q in range(n_qubits):
                    # Basic ring topology like StronglyEntanglingLayers
                    target = (q + 1) % n_qubits
                    qml.CNOT(wires=[q, target])
                    
                    # Two Qubit Noise on both wires
                    qml.DepolarizingChannel(self.p_gate_2q, wires=q)
                    qml.DepolarizingChannel(self.p_gate_2q, wires=target)
                    
                    # Thermal noise continues
                    qml.AmplitudeDamping(self.p_amp, wires=q)
                    qml.AmplitudeDamping(self.p_amp, wires=target)
                    qml.PhaseDamping(self.p_phase, wires=q)
                    qml.PhaseDamping(self.p_phase, wires=target)

            # 3. Readout Error (Bit Flip before measurement)
            # PennyLane doesn't support 'readout error' on expval directly for exact simulation easily
            # But we can model it as a BitFlip channel on all qubits right before measurement
            for q in range(n_qubits):
                qml.BitFlip(self.p_readout, wires=q)

            # Return expectations of ALL Pauli strings
            observables = []
            for s in self.pauli_strings:
                ops = []
                for idx, char in enumerate(s):
                    if char == 'X': ops.append(qml.PauliX(idx))
                    elif char == 'Y': ops.append(qml.PauliY(idx))
                    elif char == 'Z': ops.append(qml.PauliZ(idx))
                
                if not ops: 
                    observables.append(qml.expval(qml.Identity(0)))
                elif len(ops) == 1:
                    observables.append(qml.expval(ops[0]))
                else:
                    prod = ops[0]
                    for op in ops[1:]:
                        prod = prod @ op
                    observables.append(qml.expval(prod))
            
            return observables

        self.qnode = circuit
        
        # 2. Parameters (Same as ExactSIM)
        weight_shapes = {"inputs": (n_layers, n_qubits, 3)}
        self.circuit_weights = nn.Parameter(torch.rand(n_layers, n_qubits, 3, dtype=torch.float64))
        self.w = nn.Parameter((torch.randn(self.n_paulis) * 0.01).double())
        self.dim = 2**n_qubits
        self.b = nn.Parameter(torch.zeros(self.dim, dtype=torch.float64))

        # 3. Precompute Pauli Matrices (Shared logic)
        P_stack = np.array([get_pauli_tensor(s) for s in self.pauli_strings])
        self.register_buffer('P_tensor', torch.tensor(P_stack, dtype=torch.float64))
        
    def forward(self, x):
        # 1. Classical part
        x_tilde = x + self.b
        classical_features = torch.einsum('bm, kmn, bn -> bk', x_tilde, self.P_tensor, x_tilde)
        
        # 2. Quantum part (NOISY)
        quantum_expectations = self.qnode(self.circuit_weights)
        if isinstance(quantum_expectations, (list, tuple)):
            quantum_expectations = torch.stack(quantum_expectations)
        
        # 3. Combine
        combined_weights = self.w * quantum_expectations
        logits = torch.sum(classical_features * combined_weights, dim=1)
        
        return torch.sigmoid(logits)
