import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .sim_classifier import SIMClassifier
from .exact_sim_classifier import ExactSIMClassifier
from ..utils.pauli_decompose import ExtractGoldenStrings
from ..utils.noise_utils import get_depolarizing_noise_model

from src.utils.seeds import set_seed
set_seed()


def generate_fwht_pauli_strings(X, y, n_qubits, top_k=None, eta=None):
    """
    Computes Class-Conditional Covariance Difference (Delta) and extracts Pauli Strings using 
    O(N^2 log N) Fast Walsh-Hadamard Transform (FWHT).
    """
    dim = 2**n_qubits
    
    X0 = X[y == 0]
    X1 = X[y == 1]
    
    if len(X0) < 2 or len(X1) < 2:
        print("Warning: Not enough samples per class for covariance. Result will be zero.")
        R0 = np.zeros((dim, dim))
        R1 = np.zeros((dim, dim))
    else:
        # Use simple matrix multiplication for R
        R0 = (X0.T @ X0) / len(X0)
        R1 = (X1.T @ X1) / len(X1)
        
    Delta = R1 - R0
    
    # Use the blazing-fast FWHT backend
    golden_strings = ExtractGoldenStrings(Delta, n_qubits, k=top_k, eta=eta)
    
    strings = [s[0] for s in golden_strings]
    # Extract Real parts (imaginary is 0 for symmetric real Delta)
    coefs = [np.real(s[1]) for s in golden_strings]
    
    return strings, coefs


class FWHTSIMClassifier(SIMClassifier):
    """
    Unified FWHT SIM Classifier (Classical version).
    
    This model integrates the Fast Walsh-Hadamard Transform Pauli Decomposition directly into 
    the training pipeline. During `fit()`, it automatically determines the most 
    important Pauli interaction strings based on the class-conditional covariance 
    difference (Delta) in O(N^2 log N) time natively.
    
    Attributes:
        n_qubits (int): The number of qubits.
        top_k (int): Number of top Pauli strings to select.
        eta (float): Energy cutoff fraction (used if top_k is None).
        C (float): Regularization parameter for the internal classifier.
        random_state (int): Random seed.
    """
    def __init__(self, n_qubits, top_k=50, eta=None, C=1.0, random_state=None):
        super().__init__(pauli_strings=None, C=C, random_state=random_state)
        self.n_qubits = n_qubits
        self.top_k = top_k
        self.eta = eta
        self.spectral_coefs_ = []
        
    def fit(self, X, y):
        """
        Dynamically generates Pauli strings using the FWHT method,
        then fits the underlying SIM classifier.
        """
        strings, coefs = generate_fwht_pauli_strings(
            X, y, self.n_qubits, top_k=self.top_k, eta=self.eta
        )
        self.pauli_strings = strings
        self.spectral_coefs_ = coefs
        
        super().fit(X, y)
        return self


class FWHTExactSIMClassifier(nn.Module):
    """
    Unified FWHT Exact SIM Classifier (Quantum Pytorch version).
    
    This PyTorch module dynamically determines its Pauli basis during `fit()` 
    using the FWHT Pauli algorithm, then initializes and trains the 
    `ExactSIMClassifier` under the hood.
    
    Args:
        n_qubits (int): Number of qubits.
        n_layers (int): Depth of the quantum circuit.
        top_k (int): Number of top Pauli strings to select.
        eta (float): Energy cutoff fraction (used if top_k is None).
        device_name (str): PennyLane device name.
    """
    def __init__(self, n_qubits, n_layers=3, top_k=50, eta=None, device_name='default.qubit', noise_prob=0.0, pauli_strings=None, shots=None, **device_kwargs):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.top_k = top_k
        self.eta = eta
        self.device_name = device_name
        self.noise_prob = noise_prob
        
        self.pauli_strings_ = pauli_strings if pauli_strings is not None else []
        self.spectral_coefs_ = []
        self.shots = shots
        self.device_kwargs = device_kwargs
        
        self.model = None

    def fit(self, X, y, lr=0.01, epochs=50, batch_size=None, verbose=True):
        """
        Selects basis via FWHT method, initializes ExactSIMClassifier, 
        and trains it via gradient descent.
        """
        if not self.pauli_strings_:
            if verbose:
                sel_mod = f"top_k={self.top_k}" if self.top_k is not None else f"eta={self.eta}"
                print(f"Generating Fast FWHT Pauli Strings ({sel_mod})...")
            
            strings, coefs = generate_fwht_pauli_strings(
                X, y, self.n_qubits, top_k=self.top_k, eta=self.eta
            )
            
            self.pauli_strings_ = strings
            self.spectral_coefs_ = coefs
        elif verbose:
            print("Using provided Pauli strings basis.")
        
        if verbose:
            print("Initializing ExactSIMClassifier...")
            
        wires = list(range(self.n_qubits))
        noise_model = get_depolarizing_noise_model(self.noise_prob, wires=wires) if self.noise_prob > 0 else None
        
        dev_name = self.device_name
        if noise_model and 'default' in dev_name:
            dev_name = 'default.mixed'
        
        self.model = ExactSIMClassifier(
            n_qubits=self.n_qubits, 
            n_layers=self.n_layers, 
            device_name=dev_name, 
            pauli_strings=self.pauli_strings_,
            noise_model=noise_model,
            shots=self.shots,
            **self.device_kwargs
        )
        if next(self.parameters(), None) is not None:
            self.model.to(next(self.parameters()).device)
            
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        X_t = torch.tensor(X, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64)
        
        if batch_size is None:
            batch_size = len(X)
            
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if verbose:
            print("Training Model...")
            
        self.model.train()
        for ep in range(epochs):
            total_loss = 0.0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bx)
                
            avg_loss = total_loss / len(X)
            if verbose and (ep+1) % 10 == 0:
                print(f"  Epoch {ep+1}/{epochs} | Loss: {avg_loss:.4f}")
                
        return self

    def forward(self, x):
        if self.model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        return self.model(x)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float64)
            preds = (self.model(X_t) > 0.5).int().numpy()
        return preds
