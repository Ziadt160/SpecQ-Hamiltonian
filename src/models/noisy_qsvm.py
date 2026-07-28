import pennylane as qml
import numpy as np
from sklearn.svm import SVC
from ..utils.noise_utils import get_depolarizing_noise_model

class NoisyQSVM:
    def __init__(self, n_qubits, noise_prob=0.0):
        self.n_qubits = n_qubits
        self.noise_prob = noise_prob
        self.dev = qml.device("default.mixed", wires=n_qubits)
        
        def kernel_circuit(x1, x2):
            qml.AmplitudeEmbedding(features=x1, wires=range(self.n_qubits), normalize=True, pad_with=0.)
            qml.adjoint(qml.AmplitudeEmbedding)(features=x2, wires=range(self.n_qubits), normalize=True, pad_with=0.)
            return qml.probs(wires=range(self.n_qubits))
            
        self.qnode = qml.QNode(kernel_circuit, self.dev, interface="autograd")
        
        # Apply Transform
        if noise_prob > 0:
            nm = get_depolarizing_noise_model(noise_prob, wires=list(range(self.n_qubits)))
            self.qnode = qml.add_noise(self.qnode, nm)
            
        self.svm = None

    def q_kernel(self, A, B):
        return np.array([[self.qnode(a, b)[0] for b in B] for a in A])

    def fit(self, X, y):
        self.svm = SVC(kernel=self.q_kernel, probability=True)
        self.svm.fit(X, y)

    def predict_proba(self, X):
        return self.svm.predict_proba(X)
