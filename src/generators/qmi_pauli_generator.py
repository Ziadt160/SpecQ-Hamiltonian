r"""
Quadratic Mutual Information (QMI) Pauli selection.

Ranks Pauli strings by the statistical dependence between the quadratic feature
f_P(x) = x^T P x and the labels y, estimated with Parzen windows.

The score is the Cauchy-Schwarz quadratic mutual information (Principe et al.):

    QMI_CS = log( V_J * V_M / V_C^2 )

    V_J = <K_x(i,j) * K_y(i,j)>_ij              joint information potential
    V_M = <K_x(i,j)>_ij * <K_y(i,j)>_ij         product of marginals
    V_C = < (<K_x(i,j)>_j) * (<K_y(i,k)>_k) >_i cross term

QMI_CS >= 0, and equals 0 exactly when f and y are independent, so larger means
more dependent and the ranking is a plain descending sort.

Previous implementation returned -log(V_xy^2 / (V_x V_y)), which is *decreasing*
in dependence: on a controlled test it scored an independent feature 1.63, a
partially informative one 1.22 and a perfect predictor 0.27, then sorted
descending -- i.e. it selected the least informative strings. Any result
produced with the old scorer should be regarded as selecting anti-informative
interactions and rerun.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..utils.pauli_utils import generate_pauli_strings, get_pauli_tensor


def compute_quadratic_features(X, paulis):
    """
    f_P(x) = x^T P x for each sample and each Pauli matrix.

    Args:
        X (np.ndarray): (n_samples, dim) inputs.
        paulis (list of np.ndarray): (dim, dim) Hermitian matrices.

    Returns:
        np.ndarray: (n_samples, n_paulis) real features.
    """
    feats = [np.einsum('ni,ij,nj->n', X, P, X).real for P in paulis]
    return np.array(feats).T


def _feature_kernel(f, sigma=None):
    """Gaussian Parzen kernel matrix with the median heuristic for the width."""
    dist_sq = squareform(pdist(np.asarray(f).reshape(-1, 1), 'sqeuclidean'))
    if sigma is None:
        nonzero = dist_sq[dist_sq > 0]
        sigma = np.sqrt(np.median(nonzero) / 2.0) if nonzero.size else 1.0
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = 1.0
    return np.exp(-dist_sq / (2.0 * sigma ** 2))


def qmi_score(f, y, sigma=None):
    """
    Cauchy-Schwarz QMI between feature values `f` and labels `y`.

    Returns 0.0 for a degenerate feature (constant, or a single class present).
    Larger values indicate stronger dependence.
    """
    f = np.asarray(f, dtype=float).ravel()
    y = np.asarray(y).ravel()

    if f.size == 0 or np.unique(y).size < 2 or np.ptp(f) <= 1e-12:
        return 0.0

    Kx = _feature_kernel(f, sigma)
    Ky = (y[:, None] == y[None, :]).astype(float)

    V_J = float(np.mean(Kx * Ky))
    V_M = float(np.mean(Kx) * np.mean(Ky))
    V_C = float(np.mean(Kx.mean(axis=1) * Ky.mean(axis=1)))

    eps = 1e-12
    if V_J <= eps or V_M <= eps or V_C <= eps:
        return 0.0

    return max(float(np.log((V_J * V_M) / (V_C ** 2 + eps) + eps)), 0.0)


def generate_qmi_pauli_strings(X, y, n_qubits, top_k=None, skip_identity=True):
    """
    Ranks Pauli strings by QMI between x^T P x and y, most dependent first.

    Args:
        X (np.ndarray): (n_samples, 2^n_qubits) inputs.
        y (np.ndarray): labels.
        n_qubits (int): number of qubits.
        top_k (int, optional): return only the top k.
        skip_identity (bool): drop the all-identity string, whose feature is
            proportional to ||x||^2 and carries no interaction information.

    Returns:
        (sorted_strings, sorted_scores)
    """
    strings = generate_pauli_strings(n_qubits)
    if skip_identity:
        identity = 'I' * n_qubits
        strings = [s for s in strings if s != identity]

    scores = []
    for s in strings:
        f = compute_quadratic_features(X, [get_pauli_tensor(s)])[:, 0]
        scores.append(qmi_score(f, y))

    order = np.argsort(-np.asarray(scores), kind='stable')
    sorted_strings = [strings[i] for i in order]
    sorted_scores = [scores[i] for i in order]

    if top_k is not None:
        return sorted_strings[:top_k], sorted_scores[:top_k]
    return sorted_strings, sorted_scores
