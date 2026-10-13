"""
SIM classifiers that choose their own Pauli basis from the training data.

Selection happens inside ``fit()``, so it only ever sees the training split.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..generators.spectral_pauli_generator import generate_spectral_pauli_strings
from ..utils.noise_utils import get_depolarizing_noise_model
from .exact_sim_classifier import ExactSIMClassifier
from .sim_classifier import SIMClassifier


class SpectralSIMClassifier(SIMClassifier):
    """
    Unified Spectral SIM Classifier (Classical version).

    This model integrates the Spectral Pauli Generation algorithm directly into
    the training pipeline. During `fit()`, it automatically determines the most
    important Pauli interaction strings based on the class-conditional
    difference matrix (Delta).

    Attributes:
        n_qubits (int): The number of qubits (determines the dimension `2^n_qubits`).
        top_k (int): Number of top Pauli strings to select.
        C (float): Regularization parameter for the internal classifier.
        random_state (int): Random seed.
        moment (str): Delta definition, 'covariance' (default) or 'second_moment'.
    """
    def __init__(self, n_qubits, top_k=50, C=1.0, random_state=None, moment='covariance'):
        # We start with empty pauli_strings, they will be generated in fit()
        super().__init__(pauli_strings=None, C=C, random_state=random_state)
        self.n_qubits = n_qubits
        self.top_k = top_k
        self.moment = moment
        self.spectral_coefs_ = []

    def fit(self, X, y):
        """
        Dynamically generates Pauli strings using the spectral method,
        then fits the underlying SIM classifier.
        """
        strings, coefs = generate_spectral_pauli_strings(
            X, y, self.n_qubits, top_k=self.top_k, moment=self.moment)
        self.pauli_strings = strings
        # Real for a real symmetric Delta: odd-Y strings project to exactly zero.
        self.spectral_coefs_ = [np.real(c) for c in coefs]

        super().fit(X, y)
        return self


class BasisSelectingExactSIM(nn.Module):
    """
    Shared training loop for `ExactSIMClassifier` wrappers that select their basis.

    Subclasses implement `select_basis(X, y) -> (strings, coefficients)`. Binary
    classification only (labels in {0, 1}, BCE loss).

    Args:
        n_qubits (int): Number of qubits.
        n_layers (int): Depth of the quantum circuit.
        top_k (int): Number of Pauli strings to select.
        device_name (str): PennyLane device name.
        noise_prob (float): Per-gate depolarizing probability; > 0 switches a
            default.* device to default.mixed.
        pauli_strings (list of str, optional): A fixed basis. When given, no
            selection is performed.
        shots (int, optional): Finite shot budget; None means analytic.
        **device_kwargs: Forwarded to qml.device.
    """

    def __init__(self, n_qubits, n_layers=3, top_k=50, device_name='default.qubit',
                 noise_prob=0.0, pauli_strings=None, shots=None, **device_kwargs):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.top_k = top_k
        self.device_name = device_name
        self.noise_prob = noise_prob
        self.shots = shots
        self.device_kwargs = device_kwargs

        # Kept separate from the fitted basis: previously a second fit() reused
        # the first fit's selection, because the selected strings were written
        # back into the same attribute that marks a user-supplied basis.
        self.fixed_pauli_strings = pauli_strings
        self.pauli_strings_ = list(pauli_strings) if pauli_strings is not None else []
        self.spectral_coefs_ = []

        # Instantiated in fit()
        self.model = None

    def select_basis(self, X, y):
        raise NotImplementedError

    def _describe_selection(self):
        return f"Top-{self.top_k}"

    def fit(self, X, y, lr=0.01, epochs=50, batch_size=None, verbose=True):
        """
        Selects the basis, initializes ExactSIMClassifier, and trains it with Adam.
        """
        # 1. Basis selection, on the data passed to fit() only
        if self.fixed_pauli_strings is None:
            if verbose:
                print(f"Selecting {self._describe_selection()} Pauli strings...")
            self.pauli_strings_, self.spectral_coefs_ = self.select_basis(X, y)
        elif verbose:
            print("Using provided Pauli strings basis.")

        # 2. Initialize Model
        if verbose:
            print("Initializing ExactSIMClassifier...")

        wires = list(range(self.n_qubits))
        noise_model = (get_depolarizing_noise_model(self.noise_prob, wires=wires)
                       if self.noise_prob > 0 else None)
        # Noise channels need a mixed-state simulator; a real backend is left alone.
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

    def predict_proba(self, X):
        """P(y = 1 | x) for each row of X, as a NumPy array."""
        if self.model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float64)).numpy()

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


class SpectralExactSIMClassifier(BasisSelectingExactSIM):
    """
    Unified Spectral Exact SIM Classifier (Quantum PyTorch version).

    Selects the top_k strings by spectral energy |c_P| during `fit()`, then
    trains an `ExactSIMClassifier` on that basis. See `BasisSelectingExactSIM`
    for the arguments; `moment` selects the Delta definition.
    """

    def __init__(self, n_qubits, n_layers=3, top_k=50, device_name='default.qubit',
                 noise_prob=0.0, pauli_strings=None, shots=None, moment='covariance',
                 **device_kwargs):
        super().__init__(n_qubits, n_layers=n_layers, top_k=top_k, device_name=device_name,
                         noise_prob=noise_prob, pauli_strings=pauli_strings, shots=shots,
                         **device_kwargs)
        self.moment = moment

    def select_basis(self, X, y):
        strings, coefs = generate_spectral_pauli_strings(
            X, y, self.n_qubits, top_k=self.top_k, moment=self.moment)
        return strings, [np.real(c) for c in coefs]
