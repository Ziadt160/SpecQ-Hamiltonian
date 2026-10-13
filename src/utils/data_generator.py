import numpy as np

from .pauli_utils import generate_pauli_strings, get_pauli_tensor


def generate_dataset(n_samples, n_qubits, regime, random_state=42, hamiltonian_seed=123):
    """
    Generates synthetic datasets for specific physical regimes.

    Each regime labels a random unit vector x by the sign of x^T H x, where H is
    a random combination of one class of Pauli strings. A SIM classifier should
    solve a regime exactly once its basis contains the matching class.

    Args:
        n_samples (int): Number of samples.
        n_qubits (int): Number of qubits (dimension = 2^n).
        regime (str): 'linear' (Z_i), 'pairwise' (X_i X_j) or 'conditional' (Z_i X_j).
        random_state (int): Seed for X generation.
        hamiltonian_seed (int): Seed for Hamiltonian (target) generation.

    Returns:
        X (np.array): Features (n_samples, 2^n_qubits).
        y (np.array): Labels in {0, 1} (n_samples,), 1 where x^T H x > 0.
            Previously {-1, +1} via np.sign, unlike every other loader, which
            broke anything that splits classes on y == 0 / y == 1 (including
            spectral selection) and would break BCELoss.
    """
    # 1. Generate X using random_state
    rng_x = np.random.RandomState(random_state)
    dim = 2**n_qubits

    X = rng_x.randn(n_samples, dim)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # 2. Generate H using hamiltonian_seed (FIXED for both train and test)
    rng_h = np.random.RandomState(hamiltonian_seed)

    if regime == 'linear':
        # "Linear/Additive": Sum of Z_i terms (diagonal)
        H_diag = np.zeros(dim)
        for i in range(n_qubits):
            # Z_i diagonal pattern
            weight = rng_h.randn()
            term_diag = np.array([(-1)**((idx >> (n_qubits - 1 - i)) & 1) for idx in range(dim)])
            H_diag += weight * term_diag

        # Compute expectations <psi|H|psi> = sum_k |x_k|^2 * H_kk
        expectations = np.sum((X**2) * H_diag, axis=1)

    elif regime == 'pairwise':
        # "Pairwise": Sum of X_i X_j terms
        all_strs = generate_pauli_strings(n_qubits)
        xx_strs = [s for s in all_strs if s.count('X') == 2 and s.count('Z')==0 and s.count('Y')==0]

        if not xx_strs:
            xx_strs = ['XX'] if n_qubits >= 2 else ['X']

        H_mat = np.zeros((dim, dim))
        for s in xx_strs:
            w = rng_h.randn()
            H_mat += w * get_pauli_tensor(s).real

        expectations = np.einsum('ij,jk,ik->i', X, H_mat, X)

    elif regime == 'conditional':
        # "Conditional": Sum of Z_i X_j terms
        # Represents interaction between contrast (Z) and flip (X)

        all_strs = generate_pauli_strings(n_qubits)
        # Exactly one X and one Z, and no Y: the filter previously omitted the Y
        # condition, so strings such as 'XZYY' (one X, one Z, two Ys -> a real
        # matrix that survives .real) were mixed into what is documented as a
        # pure Z_i X_j regime.
        zx_strs = [s for s in all_strs
                   if s.count('X') == 1 and s.count('Z') == 1 and s.count('Y') == 0]

        H_mat = np.zeros((dim, dim))
        for s in zx_strs:
            w = rng_h.randn()
            H_mat += w * get_pauli_tensor(s).real

        expectations = np.einsum('ij,jk,ik->i', X, H_mat, X)

    else:
        raise ValueError(f"Unknown regime: {regime}")

    y = (expectations > 0).astype(int)
    return X, y
