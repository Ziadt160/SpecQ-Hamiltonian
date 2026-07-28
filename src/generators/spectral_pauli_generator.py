"""
Spectral Moment Pauli Generator (Algorithm 3).

Ranks Pauli strings by how much they contribute to the class-conditional
difference matrix Delta, decomposed in the Pauli basis:

    Delta = sum_P c_P P        c_P = (1/2^n) Tr(Delta P)

Strings with large |c_P| are the observables that best separate the classes.

Two definitions of Delta are supported (see `moment`):

  'second_moment'  Delta = E[xx^T | y=1] - E[xx^T | y=0]
  'covariance'     Delta = Cov(x | y=1)  - Cov(x | y=0)

These coincide only when both class-conditional means are zero, which does not
hold for the datasets here. The Spectral Pauli Pruning paper (Eq. 4) specifies
*covariance*; the original implementation computed the *second moment*.

The default is 'covariance', matching the paper. That choice is made on
principle, NOT on measured accuracy:

  * Eq. 4 of the Spectral Pauli Pruning manuscript specifies covariance;
  * the second moment conflates class-mean separation with covariance
    structure, which makes the selected basis harder to interpret.

On accuracy the two are indistinguishable. A controlled comparison on E. Coli
N=6 (5 seeds, identical pipeline, only Delta varying) gives covariance +0.040
at k=8, +0.074 at k=16, but -0.041 at k=128 and -0.003 at the full basis -- with
per-seed standard deviations of 0.02-0.08, i.e. as large as the gaps.
experiments/experiment_delta_definition.py agrees: 60 configuration wins to 58,
which is a coin flip. Do not claim one definition generalises better.

The choice does change *which* strings are selected -- only 21-25% overlap in
the top-16 on E. Coli, converging to 94% by k=128 -- so results are not
comparable across settings. Pass moment='second_moment' to reproduce older runs.
"""
import numpy as np

from ..utils.pauli_utils import generate_pauli_strings, get_pauli_tensor

VALID_MOMENTS = ('second_moment', 'covariance')


def compute_delta(X, y, moment='covariance'):
    """
    Class-conditional difference matrix.

    Args:
        X (np.ndarray): (n_samples, dim) inputs.
        y (np.ndarray): binary labels.
        moment (str): 'second_moment' or 'covariance'.

    Returns:
        np.ndarray: (dim, dim) real symmetric matrix.
    """
    if moment not in VALID_MOMENTS:
        raise ValueError(f"moment must be one of {VALID_MOMENTS}, got {moment!r}")

    dim = X.shape[1]
    X0, X1 = X[y == 0], X[y == 1]

    if len(X0) < 2 or len(X1) < 2:
        print("Warning: Not enough samples per class to estimate Delta.")
        return np.zeros((dim, dim))

    if moment == 'second_moment':
        R0 = (X0.T @ X0) / len(X0)
        R1 = (X1.T @ X1) / len(X1)
    else:  # covariance -- subtract the class means first (paper Eq. 4)
        R0 = np.cov(X0, rowvar=False)
        R1 = np.cov(X1, rowvar=False)

    return R1 - R0


def pauli_coefficients(Delta, pauli_strings):
    """
    Projects Delta onto each Pauli string: c_P = (1/2^n) Tr(Delta P).

    Uses a single vectorised contraction rather than 4^n explicit matrix
    products. Tr(Delta P) = sum_{i,j} Delta[i,j] P[j,i], so the whole stack
    costs O(4^n * 2^2n) instead of O(4^n * 2^3n) for the matmul-then-trace form.
    Values are bit-identical to np.trace(Delta @ P).

    Args:
        Delta (np.ndarray): (dim, dim).
        pauli_strings (list of str): strings to project onto.

    Returns:
        np.ndarray: complex coefficients, one per string.
    """
    dim = Delta.shape[0]
    P_stack = np.array([get_pauli_tensor(s) for s in pauli_strings])
    return np.einsum('ij,kji->k', Delta, P_stack) / dim


def generate_spectral_pauli_strings(X, y, n_qubits, top_k=None, moment='covariance'):
    """
    Algorithm 3: rank Pauli strings by spectral energy |c_P|.

    Args:
        X (np.ndarray): (n_samples, 2^n_qubits) inputs.
        y (np.ndarray): binary labels.
        n_qubits (int): number of qubits.
        top_k (int, optional): return only the top k strings.
        moment (str): 'covariance' (default, paper Eq. 4) or 'second_moment' (legacy).

    Returns:
        If top_k is None: (strings, coefficients, magnitudes)
        Otherwise:        (strings, coefficients)
    """
    Delta = compute_delta(X, y, moment=moment)
    all_strings = generate_pauli_strings(n_qubits)

    coefs = pauli_coefficients(Delta, all_strings)
    mags = np.abs(coefs)

    order = np.argsort(-mags, kind='stable')
    sorted_strings = [all_strings[i] for i in order]
    sorted_coefs = [coefs[i] for i in order]
    sorted_mags = [mags[i] for i in order]

    if top_k is not None:
        return sorted_strings[:top_k], sorted_coefs[:top_k]

    return sorted_strings, sorted_coefs, sorted_mags


def get_adaptive_spectral_paulis(X, y, n_qubits, eta=0.95, moment='covariance'):
    """
    Selects the smallest Pauli set whose cumulative spectral energy reaches eta
    (paper Eq. 7).

    Args:
        eta (float): energy threshold in (0, 1].
        moment (str): see generate_spectral_pauli_strings.

    Returns:
        (strings, coefficients, k_cutoff)
    """
    strings, coefs, mags = generate_spectral_pauli_strings(
        X, y, n_qubits, top_k=None, moment=moment
    )

    total_energy = float(np.sum(mags))
    if total_energy <= 0:
        # Delta is identically zero (degenerate class); nothing to rank.
        return [], [], 0

    cumulative = np.cumsum(mags) / total_energy
    k_cutoff = int(np.searchsorted(cumulative, eta) + 1)
    k_cutoff = min(k_cutoff, len(strings))

    return strings[:k_cutoff], coefs[:k_cutoff], k_cutoff


def is_dead_string(s):
    """
    True if x^T P x is identically zero for every real-valued x.

    A Pauli string with an odd number of Y factors is a purely imaginary
    Hermitian matrix, so its real quadratic form vanishes exactly. Measuring
    such an observable returns no information about real-valued inputs.
    The fraction of such strings is (4^n - 2^n) / (2 * 4^n) -> 50% as n grows.
    """
    return s.count('Y') % 2 == 1


def count_dead_strings(n_qubits):
    """Number of identically-zero observables in the full n-qubit Pauli basis."""
    return (4 ** n_qubits - 2 ** n_qubits) // 2
