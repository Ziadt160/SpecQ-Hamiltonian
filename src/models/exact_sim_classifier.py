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
    def __init__(self, n_qubits, n_layers=3, device_name='default.qubit', pauli_strings=None):
        super().__init__()
        self.n_qubits = n_qubits
        if pauli_strings is not None:
            self.pauli_strings = pauli_strings
        else:
            self.pauli_strings = generate_pauli_strings(n_qubits)
        self.n_paulis = len(self.pauli_strings)
        
        # 1. Quantum Device & Circuit
        self.dev = qml.device(device_name, wires=n_qubits)
        
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
        
        # 2. Parameters
        # Circuit weights: shape for StronglyEntangling is (n_layers, n_qubits, 3)
        weight_shapes = {"inputs": (n_layers, n_qubits, 3)}
        # We start with random weights
        self.circuit_weights = nn.Parameter(torch.rand(n_layers, n_qubits, 3, dtype=torch.float64))
        
        self.w = nn.Parameter((torch.randn(self.n_paulis) * 0.01).double())
        
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
        x: (batch_size, dim)
        """
        # 1. Bias: x_tilde = x + b
        # Paper says: x_tilde = x + b. Then re-normalize? 
        # Paper Eq 5: x_tilde = 1/s sum x_i + b. For single sample: x + b.
        # Usually quantum states need L2 norm=1. 
        # But SIM formulation handles non-normalized inputs via the quadratic form.
        # We'll stick to x + b.
        x_tilde = x + self.b
        
        # 2. Classical Feature Map: phi_j = x_tilde^T P_j x_tilde
        # Efficient computation:
        # P_tensor: (K, D, D)
        # x_tilde: (B, D)
        # We want Result: (B, K) where R_bj = x_b^T P_j x_b
        
        # Reshape for broadcast/einsum
        # P x^T -> (K, D, D) @ (D, B)^T ?? No.
        # x P x^T.
        
        # Option A: Loop (Bad)
        # Option B: Einsum 'bik, kjl, bil -> bj' ? 
        # Let's align dimensions:
        # x_tilde: (B, D)
        # P_tensor: (K, D, D)
        # Term: sum_{m,n} x[b,m] * P[k,m,n] * x[b,n]
        # Einsum: 'bm, kmn, bn -> bk'
        
        # Note: x_tilde might become huge if batch is large. 
        # For N=4 (D=16), K=256, it's tiny. Safe.
        
        classical_features = torch.einsum('bm, kmn, bn -> bk', x_tilde, self.P_tensor, x_tilde)
        
        # 3. Quantum Expectations: E_j = <psi | P_j | psi>
        # This depends only on circuit_weights, so it's constant for the batch!
        # Returns list of Tensors or stacked Tensor
        quantum_expectations = self.qnode(self.circuit_weights)
        # If qnode returns tuple/list, stack them
        if isinstance(quantum_expectations, (list, tuple)):
            quantum_expectations = torch.stack(quantum_expectations) # (K,)
        
        # 4. Combine (Eq 9)
        # f = sum_j ( Classical_j * w_j * Quantum_j )
        # Dimensions: (B, K) * (K,) * (K,) -> Sum over K -> (B,)
        
        combined_weights = self.w * quantum_expectations # (K,)
        logits = torch.sum(classical_features * combined_weights, dim=1) # (B,)
        
        # 5. Sigmoid (Output probability)
        return torch.sigmoid(logits)

