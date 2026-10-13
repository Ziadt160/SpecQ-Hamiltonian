"""
SIM classifiers whose basis is selected with a Fast Walsh-Hadamard Pauli decomposition.

Selection matches `spectral_sim_classifier` -- strings ranked by |c_P| for the
same Delta -- but the decomposition costs O(4^n * n) via the FWHT instead of the
O(4^n * 4^n) dense projection, which matters from n ~ 8 upward.
"""
import numpy as np

from ..generators.spectral_pauli_generator import compute_delta
from ..utils.pauli_decompose import extract_golden_strings
from .sim_classifier import SIMClassifier
from .spectral_sim_classifier import BasisSelectingExactSIM


def generate_fwht_pauli_strings(X, y, n_qubits, top_k=None, eta=None, moment='covariance'):
    """
    Computes the class-conditional difference matrix Delta and ranks its Pauli
    coefficients using the O(N^2 log N) Fast Walsh-Hadamard Transform (FWHT).

    Previously Delta was always the second moment E[xx^T|y=1] - E[xx^T|y=0],
    so this path disagreed with `generate_spectral_pauli_strings` after its
    default moved to the covariance (paper Eq. 4). Both now share
    `compute_delta`.

    Args:
        top_k (int, optional): number of strings to keep.
        eta (float, optional): used when top_k is None. Note the FWHT path
            accumulates *squared* magnitudes |c_P|^2 (Frobenius energy), whereas
            `get_adaptive_spectral_paulis` accumulates |c_P|; the same eta
            therefore selects a different number of strings.
        moment (str): 'covariance' (default) or 'second_moment'.

    Returns:
        (strings, coefs) with real coefficients, largest |c_P| first.
    """
    Delta = compute_delta(X, y, moment=moment)
    golden_strings = extract_golden_strings(Delta, n_qubits, k=top_k, eta=eta)

    strings = [s[0] for s in golden_strings]
    # Real parts: the imaginary part is zero for a real symmetric Delta
    coefs = [np.real(s[1]) for s in golden_strings]

    return strings, coefs


class FWHTSIMClassifier(SIMClassifier):
    """
    Unified FWHT SIM Classifier (Classical version).

    This model integrates the Fast Walsh-Hadamard Transform Pauli Decomposition directly into
    the training pipeline. During `fit()`, it automatically determines the most
    important Pauli interaction strings based on the class-conditional
    difference matrix (Delta) in O(N^2 log N) time.

    Attributes:
        n_qubits (int): The number of qubits.
        top_k (int): Number of top Pauli strings to select.
        eta (float): Energy cutoff fraction (used if top_k is None).
        C (float): Regularization parameter for the internal classifier.
        random_state (int): Random seed.
        moment (str): Delta definition, 'covariance' (default) or 'second_moment'.
    """
    def __init__(self, n_qubits, top_k=50, eta=None, C=1.0, random_state=None,
                 moment='covariance'):
        super().__init__(pauli_strings=None, C=C, random_state=random_state)
        self.n_qubits = n_qubits
        self.top_k = top_k
        self.eta = eta
        self.moment = moment
        self.spectral_coefs_ = []

    def fit(self, X, y):
        """
        Dynamically generates Pauli strings using the FWHT method,
        then fits the underlying SIM classifier.
        """
        strings, coefs = generate_fwht_pauli_strings(
            X, y, self.n_qubits, top_k=self.top_k, eta=self.eta, moment=self.moment
        )
        self.pauli_strings = strings
        self.spectral_coefs_ = coefs

        super().fit(X, y)
        return self


class FWHTExactSIMClassifier(BasisSelectingExactSIM):
    """
    Unified FWHT Exact SIM Classifier (Quantum PyTorch version).

    Selects its basis with `generate_fwht_pauli_strings` during `fit()`, then
    trains an `ExactSIMClassifier` on it. See `BasisSelectingExactSIM` for the
    shared arguments.

    Args:
        eta (float): Energy cutoff fraction (used if top_k is None).
        moment (str): Delta definition, 'covariance' (default) or 'second_moment'.
    """
    def __init__(self, n_qubits, n_layers=3, top_k=50, eta=None, device_name='default.qubit',
                 noise_prob=0.0, pauli_strings=None, shots=None, moment='covariance',
                 **device_kwargs):
        super().__init__(n_qubits, n_layers=n_layers, top_k=top_k, device_name=device_name,
                         noise_prob=noise_prob, pauli_strings=pauli_strings, shots=shots,
                         **device_kwargs)
        self.eta = eta
        self.moment = moment

    def _describe_selection(self):
        return f"FWHT top_k={self.top_k}" if self.top_k is not None else f"FWHT eta={self.eta}"

    def select_basis(self, X, y):
        return generate_fwht_pauli_strings(
            X, y, self.n_qubits, top_k=self.top_k, eta=self.eta, moment=self.moment)
