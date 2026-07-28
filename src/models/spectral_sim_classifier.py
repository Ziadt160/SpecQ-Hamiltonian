import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .sim_classifier import SIMClassifier
from .exact_sim_classifier import ExactSIMClassifier
from ..generators.spectral_pauli_generator import generate_spectral_pauli_strings
from ..utils.noise_utils import get_depolarizing_noise_model

from src.utils.seeds import set_seed
set_seed()


class SpectralSIMClassifier(SIMClassifier):
    """
    Unified Spectral SIM Classifier (Classical version).
    
    This model integrates the Spectral Pauli Generation algorithm directly into 
    the training pipeline. During `fit()`, it automatically determines the most 
    important Pauli interaction strings based on the class-conditional covariance 
    difference (Delta).
    
    Attributes:
        n_qubits (int): The number of qubits (determines the dimension `2^n_qubits`).
        top_k (int): Number of top Pauli strings to select.
        C (float): Regularization parameter for the internal classifier.
        random_state (int): Random seed.
    """
    def __init__(self, n_qubits, top_k=50, C=1.0, random_state=None):
        # We start with empty pauli_strings, they will be generated in fit()
        super().__init__(pauli_strings=None, C=C, random_state=random_state)
        self.n_qubits = n_qubits
        self.top_k = top_k
        self.spectral_coefs_ = []
        
    def fit(self, X, y):
        """
        Dynamically generates Pauli strings using the spectral method,
        then fits the underlying SIM classifier.
        """
        # 1. Spectral Generation
        # Note: generate_spectral_pauli_strings returns (strings, coefs) if top_k is NOT None
        strings, coefs = generate_spectral_pauli_strings(X, y, self.n_qubits, top_k=self.top_k)
        self.pauli_strings = strings
        self.spectral_coefs_ = coefs
        
        # 2. Fit underlying SIM model
        super().fit(X, y)
        return self


class SpectralExactSIMClassifier(nn.Module):
    """
    Unified Spectral Exact SIM Classifier (Quantum Pytorch version).
    
    This PyTorch module dynamically determines its Pauli basis during `fit()` 
    using the Spectral Pauli algorithm, then initializes and trains the 
    `ExactSIMClassifier` under the hood.
    
    Args:
        n_qubits (int): Number of qubits.
        n_layers (int): Depth of the quantum circuit.
        top_k (int): Number of top Pauli strings to select.
        device_name (str): PennyLane device name.
    """
    def __init__(self, n_qubits, n_layers=3, top_k=50, device_name='default.qubit', noise_prob=0.0, pauli_strings=None, shots=None, **device_kwargs):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.top_k = top_k
        self.device_name = device_name
        self.noise_prob = noise_prob
        
        self.pauli_strings_ = pauli_strings if pauli_strings is not None else []
        self.spectral_coefs_ = []
        self.shots = shots
        self.device_kwargs = device_kwargs
        
        # Instantiated in fit()
        self.model = None

    def fit(self, X, y, lr=0.01, epochs=50, batch_size=None, verbose=True):
        """
        Selects basis via Spectral method, initializes ExactSIMClassifier, 
        and trains it via gradient descent.
        """
        # 1. Spectral Selection
        if not self.pauli_strings_:
            if verbose:
                print(f"Generating Top-{self.top_k} Spectral Pauli Strings...")
            # Note: generate_spectral_pauli_strings returns (strings, coefs) if top_k is NOT None
            strings, coefs = generate_spectral_pauli_strings(X, y, self.n_qubits, top_k=self.top_k)
            
            self.pauli_strings_ = strings
            self.spectral_coefs_ = coefs
        elif verbose:
            print("Using provided Pauli strings basis.")
        
        # 2. Initialize Model
        if verbose:
            print("Initializing ExactSIMClassifier...")
            
        wires = list(range(self.n_qubits))
        noise_model = get_depolarizing_noise_model(self.noise_prob, wires=wires) if self.noise_prob > 0 else None
        # For noise simulation, we MUST use a mixed state device if noise_model is provided
        # But if it's a real QPU (e.g. qiskit.ibmq), we don't force 'default.mixed'
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
        # Move inner model to the same device if applicable
        if next(self.parameters(), None) is not None:
            self.model.to(next(self.parameters()).device)
            
        # 3. Train
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
