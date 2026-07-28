import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from ..utils.pauli_utils import generate_pauli_strings, get_pauli_tensor

class ExactSIMClassifier(nn.Module):
    """
    Exact (Flipped) SIM Classifier.
    
    Implements a hybrid quantum-classical classifier where the quantum state acts
    as a variational filter for classical quadratic forms.
    
    The decision function follows Equation 9:
    $$f(x) = \sigma( \sum_j (x^T P_j x) \cdot w_j \cdot \langle\psi_\theta | P_j | \psi_\theta\rangle )$$
    
    Parameters:
        n_qubits (int): Dimensions of the Hilbert space ($2^N$).
        n_layers (int): Depth of the strongly entangling ansatz.
        pauli_strings (list of str): The interaction basis.
    """
    def __init__(self, n_qubits, n_layers=3, device_name='default.qubit', pauli_strings=None, noise_model=None, shots=None, num_classes=1, **device_kwargs):
        super().__init__()
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.noise_model = noise_model
        if pauli_strings is not None:
            self.pauli_strings = pauli_strings
        else:
            self.pauli_strings = generate_pauli_strings(n_qubits)
        self.n_paulis = len(self.pauli_strings)
        
        # 1. Quantum Device & Circuit
        self.dev = qml.device(device_name, wires=n_qubits, shots=shots, **device_kwargs)
        
        # Define QNode
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs):
            # inputs is (n_layers, n_qubits, 3) for StronglyEntangling
            qml.StronglyEntanglingLayers(weights=inputs, wires=range(n_qubits))
            
            # Return expectations of ALL Pauli strings
            # Note: For N=4, 256 observables is manageable.
            # We map strings like "IXYZ" to qml.PauliX(1) @ qml.PauliY(2) ...
            observables = []
            for s in self.pauli_strings:
                ops = []
                for idx, char in enumerate(s):
                    if char == 'X': ops.append(qml.PauliX(idx))
                    elif char == 'Y': ops.append(qml.PauliY(idx))
                    elif char == 'Z': ops.append(qml.PauliZ(idx))
                
                if not ops: # Identity "IIII"
                    # Identity expectation is always 1. We handle consistent return type
                    # qml.Identity(0) expectation is 1
                    observables.append(qml.expval(qml.Identity(0)))
                elif len(ops) == 1:
                    observables.append(qml.expval(ops[0]))
                else:
                    # Tensor product
                    prod = ops[0]
                    for op in ops[1:]:
                        prod = prod @ op
                    observables.append(qml.expval(prod))
            
            return observables

        self.qnode = circuit
        
        # Professional NISQ Simulation: Apply noise transform if model provided
        if self.noise_model is not None:
            self.qnode = qml.add_noise(self.qnode, self.noise_model)
        
        # 2. Parameters
        # Circuit weights: shape for StronglyEntangling is (n_layers, n_qubits, 3)
        weight_shapes = {"inputs": (n_layers, n_qubits, 3)}
        # We start with random weights
        self.circuit_weights = nn.Parameter(torch.rand(n_layers, n_qubits, 3, dtype=torch.float64))
        
        self.w = nn.Parameter((torch.randn(self.num_classes, self.n_paulis) * 0.01).double())
        
        # Input Bias b (added to embedding)
        self.dim = 2**n_qubits
        self.b = nn.Parameter(torch.zeros(self.dim, dtype=torch.float64))

        # 3. Precompute Pauli Matrices for classical quadratic form
        # We use standard numpy matrices, convert to Torch tensors
        # Cache them? For N=4, 256 matrices of 16x16.
        # It's better to process batch-wise.
        # x^T P x.
        # We can assemble a Tensor of shape (n_paulis, dim, dim)
        P_stack = np.array([get_pauli_tensor(s) for s in self.pauli_strings])
        self.register_buffer('P_tensor', torch.tensor(P_stack, dtype=torch.float64)) # (256, 16, 16)
        
    def forward(self, x):
        """
        x: (batch_size, dim) or (batch_size, seq_len, dim)
        """
        # Eq 5: Sequence Mean-Pooling if input is 3D
        if x.dim() == 3:
            x = x.mean(dim=1)
            
        # 1. Bias: x_tilde = x + b
        x_tilde = x + self.b
        
        # 2. Classical Feature Map: phi_j = x_tilde^T P_j x_tilde
        # Including Eq 7 normalization scaling factor: 1 / 2^n
        classical_features = torch.einsum('bm, kmn, bn -> bk', x_tilde, self.P_tensor, x_tilde) * (1.0 / (2**self.n_qubits))
        
        # 3. Quantum Expectations: E_j = <psi | P_j | psi>
        quantum_expectations = self.qnode(self.circuit_weights)
        if isinstance(quantum_expectations, (list, tuple)):
            quantum_expectations = torch.stack(quantum_expectations) # (K,)
        
        # 4. Combine (Eq 9)
        # f = sum_j ( Classical_j * w_j * Quantum_j )
        # Dimensions: (B, K) * (C, K) * (K,) -> Sum over K -> (B, C)
        logits = torch.einsum('bk,ck,k->bc', classical_features, self.w, quantum_expectations)
        
        # 5. Output Normalization
        if self.num_classes == 1:
            return torch.sigmoid(logits.squeeze(1)) # Keep it binary (B,)
        return logits # Multi-class raw logits (B, C)

