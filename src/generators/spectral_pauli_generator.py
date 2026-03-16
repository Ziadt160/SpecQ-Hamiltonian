import numpy as np
import scipy.linalg
from ..utils.pauli_utils import generate_pauli_strings, get_pauli_tensor

def generate_spectral_pauli_strings(X, y, n_qubits, top_k=None):
    """
    Algorithm 3: Spectral Moment Pauli Generator.
    
    Steps:
    1. Compute class-conditional covariance difference Delta = Cov(X|y=1) - Cov(X, y=0).
    2. Decompose Delta into the Pauli basis.
       Delta = sum_k c_k P_k
       c_k = (1/2^N) * Tr(Delta * P_k)
    3. Return Pauli strings P_k sorted by magnitude |c_k|.
    
    Args:
        X (np.ndarray): Input features (N_samples, 2^n_qubits).
        y (np.ndarray): Labels (0, 1).
        n_qubits (int): Number of qubits.
        top_k (int, optional): Number of top strings to return. If None, returns all non-zero.
        
    Returns:
        list of str: Sorted Pauli strings.
        list of float: Coefficients (importance).
    """
    dim = 2**n_qubits
    
    # 1. Compute Covariances
    X0 = X[y == 0]
    X1 = X[y == 1]
    
    # Check if empty (shouldn't happen in exp)
    if len(X0) < 2 or len(X1) < 2:
        print("Warning: Not enough samples per class for covariance.")
        cov0 = np.zeros((dim, dim))
        cov1 = np.zeros((dim, dim))
    else:
        # Use simple matrix multiplication for R
        # X is (N, dim)
        R0 = (X0.T @ X0) / len(X0)
        R1 = (X1.T @ X1) / len(X1)
        
    Delta = R1 - R0
    
    # 2. Decompose into Pauli Basis
    # Generate all candidate strings
    # For N=6 (4096), this loop is fine (4096 matmuls of 64x64).
    # For N=10, this would be intractable. But we are working with N=4, 6.
    
    all_strings = generate_pauli_strings(n_qubits)
    results = []
    
    # Pre-factor for trace
    factor = 1.0 / dim
    
    for s in all_strings:
        P = get_pauli_tensor(s) # (dim, dim) complex
        # Tr(Delta * P). Delta is real, P is Hermiitan. 
        # Result should be real effectively (imag part cancels or is 0).
        val = np.trace(Delta @ P) * factor
        mag = np.abs(val)
        
        results.append((s, mag, val))
        
    # 3. Sort by magnitude
    results.sort(key=lambda x: x[1], reverse=True)
    
    sorted_strings = [r[0] for r in results]
    coefficients = [r[2] for r in results]
    magnitudes = [r[1] for r in results]
    
    if top_k is not None:
        return sorted_strings[:top_k], coefficients[:top_k]
        
    return sorted_strings, coefficients, magnitudes

def get_adaptive_spectral_paulis(X, y, n_qubits, eta=0.95):
    """
    Selects Pauli strings using Spectral Energy Cutoff.
    E(k) = sum(|c_i|) / sum_total(|c_j|) >= eta
    
    Args:
        eta (float): Energy threshold (0.0 to 1.0).
        
    Returns:
        list, list, int: strings, coefficients, k_cutoff
    """
    strings, coefs, mags = generate_spectral_pauli_strings(X, y, n_qubits, top_k=None)
    
    total_energy = sum(mags)
    current_energy = 0.0
    k_cutoff = 0
    
    for i, m in enumerate(mags):
        current_energy += m
        if current_energy / total_energy >= eta:
            k_cutoff = i + 1
            break
            
    return strings[:k_cutoff], coefs[:k_cutoff], k_cutoff

