import numpy as np
from src.pauli_utils import get_pauli_tensor, generate_pauli_strings

def generate_dataset(n_samples, n_qubits, regime, random_state=42, hamiltonian_seed=123):
    """
    Generates synthetic datasets for specific physical regimes.
    
    Args:
        n_samples (int): Number of samples.
        n_qubits (int): Number of qubits (dimension = 2^n).
        regime (str): 'linear', 'pairwise', or 'conditional'.
        random_state (int): Seed for X generation.
        hamiltonian_seed (int): Seed for Hamiltonian (target) generation.
        
    Returns:
        X (np.array): Features (n_samples, 2^n_qubits).
        y (np.array): Labels (n_samples,).
    """
    # 1. Generate X using random_state
    rng_x = np.random.RandomState(random_state)
    dim = 2**n_qubits
    
    X = rng_x.randn(n_samples, dim)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # 2. Generate H using hamiltonian_seed (FIXED for both train and test)
    rng_h = np.random.RandomState(hamiltonian_seed)
    
    y = np.zeros(n_samples)
    
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
        y = np.sign(expectations)

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
        y = np.sign(expectations)

    elif regime == 'conditional':
        # "Conditional": Sum of Z_i X_j terms
        # Represents interaction between contrast (Z) and flip (X)
        
        all_strs = generate_pauli_strings(n_qubits)
        zx_strs = [s for s in all_strs if s.count('X') == 1 and s.count('Z') == 1]
         
        H_mat = np.zeros((dim, dim))
        for s in zx_strs:
            w = rng_h.randn()
            H_mat += w * get_pauli_tensor(s).real
            
        expectations = np.einsum('ij,jk,ik->i', X, H_mat, X)
        y = np.sign(expectations)
        
    else:
        raise ValueError(f"Unknown regime: {regime}")
        
    return X, y
