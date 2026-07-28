import torch
import torch.nn as nn
import pennylane as qml
from ..utils.noise_utils import get_depolarizing_noise_model

class NoisyVQC(nn.Module):
    def __init__(self, n_qubits, layers=2, variant='StronglyEntangling', noise_prob=0.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.variant = variant
        
        # Professional approach: Default to mixed state if noise is suspected
        self.dev = qml.device('default.mixed', wires=n_qubits)
        
        if variant == 'StronglyEntangling':
            weight_shape = (layers, n_qubits, 3)
        elif variant == 'BasicEntangling':
            weight_shape = (layers, n_qubits)
            
        self.weights = nn.Parameter(torch.randn(weight_shape, dtype=torch.float64))
        
        def circuit(inputs, w):
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.)
            if variant == 'StronglyEntangling':
                qml.StronglyEntanglingLayers(weights=w, wires=range(n_qubits))
            elif variant == 'BasicEntangling':
                qml.BasicEntanglerLayers(weights=w, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))
            
        self.qnode = qml.QNode(circuit, self.dev, interface='torch')
        
        # Apply Transform
        if noise_prob > 0:
            nm = get_depolarizing_noise_model(noise_prob, wires=list(range(n_qubits)))
            self.qnode = qml.add_noise(self.qnode, nm)

    def forward(self, x):
        res = []
        for xi in x:
            out = self.qnode(xi, self.weights)
            res.append(out)
        return (torch.stack(res) + 1.0) / 2.0
