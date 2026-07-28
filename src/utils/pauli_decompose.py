"""
Golden Pauli Strings Extraction via Direct FWHT Pauli Decomposition
Implementation based on Georges et al., arXiv:2408.06206v4.

This module provides a scalable, vectorized implementation of scalar Pauli decomposition 
utilizing Fast Walsh-Hadamard Transforms (FWHT) achieving O(N^2 log N) complexity 
overall with an O(1) memory overhead mapping.

Both NumPy (CPU) and PyTorch (GPU accelerated) backends are fully supported.
"""

import numpy as np
try:
    import torch
except ImportError:
    torch = None

def pauli_index_to_string(r: int, s: int, n: int) -> str:
    """
    Translates symplectic indices (r, s) to a standard Pauli string of length n.
    Pauli elements map as: I=(0,0), X=(1,0), Y=(1,1), Z=(0,1)
    
    Returns the string where P_{n-1} (MSB) is the leftmost character, 
    matching standard quantum computing conventions (e.g. Qiskit/OpenFermion string builders).
    """
    out = []
    # From most significant bit to least significant bit
    for j in range(n - 1, -1, -1):
        rj = (r >> j) & 1
        sj = (s >> j) & 1
        if rj == 0 and sj == 0:
            out.append('I')
        elif rj == 1 and sj == 0:
            out.append('X')
        elif rj == 1 and sj == 1:
            out.append('Y')
        else: # rj == 0 and sj == 1
            out.append('Z')
    return "".join(out)

def _fwht_axis1_numpy(A: np.ndarray) -> np.ndarray:
    """
    In-place (up to slice copies) 1D iterative Fast Walsh-Hadamard Transform 
    along the columns (axis=1) of A using broadcasted reshapes.
    """
    N = A.shape[1]
    h = 1
    A = A.copy()
    while h < N:
        # Reshape to explicitly align elements that are distance 'h' apart
        shape = (A.shape[0], N // (2 * h), 2, h)
        A_view = A.reshape(shape)
        
        # Parallel butterfly evaluations
        x = A_view[:, :, 0, :].copy()
        y = A_view[:, :, 1, :].copy()
        A_view[:, :, 0, :] = x + y
        A_view[:, :, 1, :] = x - y
        h *= 2
    return A

def _extract_golden_numpy(Delta_hat: np.ndarray, n: int, eta: float = None, k: int = None):
    N = 1 << n
    
    # 1. Symplectic XOR Transformation: A_perm[r, q] = A[r \oplus q, q]
    r = np.arange(N)[:, None]
    q = np.arange(N)[None, :]
    A_perm = Delta_hat[r ^ q, q]
    
    # 2. FWHT Transformation: A_fwht[r, s] = sum_q A_perm[r, q] * H_n[q, s]
    A_fwht = _fwht_axis1_numpy(A_perm)
    
    # 3. Phase Correction: a_{r,s} = A_fwht * (-i)^{|r ^ s|} / N
    s = np.arange(N)[None, :]
    r_and_s = r & s
    
    # Secure Popcount extraction
    omega = np.zeros_like(r_and_s, dtype=int)
    for i in range(n):
        omega += (r_and_s >> i) & 1
        
    # Valid Phase mapping lookup (avoid precision issues with floats handling complex powers)
    phase_lookup = np.array([1, -1j, -1, 1j], dtype=np.complex128)
    phases = phase_lookup[omega % 4] / N
    
    # Exact coefficients c_{r, s}
    c_p = A_fwht * phases
    c_p_flat = c_p.ravel()
    
    # 4. Sorting & Thresholding Selection Logic
    mags_sq = np.abs(c_p_flat)**2
    sorted_idx = np.argsort(mags_sq)[::-1]
    
    if k is not None:
        selected_idx = sorted_idx[:k]
    elif eta is not None:
        # Convert eta target mathematically:
        # Summing absolute squares of all coefficients corresponds exactly
        # to the traced energy of the full matrix multiplied by fraction eta.
        csum = np.cumsum(mags_sq[sorted_idx])
        target = eta * csum[-1] 
        num_elements = np.searchsorted(csum, target, side='left') + 1
        num_elements = min(num_elements, len(csum))
        selected_idx = sorted_idx[:num_elements]
    else:
        selected_idx = sorted_idx
        
    # Rebuild strings
    results = []
    for idx in selected_idx:
        r_val = idx // N
        s_val = idx % N
        coef = c_p_flat[idx]
        string = pauli_index_to_string(r_val, s_val, n)
        results.append((string, coef))
        
    return results

def _fwht_axis1_torch(A: 'torch.Tensor') -> 'torch.Tensor':
    """
    Fast PyTorch FWHT along columns (axis=1) optimized for GPU broadcasting.
    """
    N = A.shape[1]
    h = 1
    A = A.clone()
    while h < N:
        A_view = A.view(A.shape[0], N // (2 * h), 2, h)
        x = A_view[:, :, 0, :].clone()
        y = A_view[:, :, 1, :].clone()
        A_view[:, :, 0, :] = x + y
        A_view[:, :, 1, :] = x - y
        h *= 2
    return A

def _extract_golden_torch(Delta_hat: 'torch.Tensor', n: int, eta: float = None, k: int = None):
    N = 1 << n
    device = Delta_hat.device
    
    # 1. GPU Optimized XOR Transform
    r = torch.arange(N, device=device).unsqueeze(1)
    q = torch.arange(N, device=device).unsqueeze(0)
    A_perm = Delta_hat[r ^ q, q]
    
    # 2. FWHT
    A_fwht = _fwht_axis1_torch(A_perm)
    
    # 3. Phase corrections natively executing on PyTorch Kernels
    s = torch.arange(N, device=device).unsqueeze(0)
    r_and_s = r & s
    
    omega = torch.zeros_like(r_and_s, dtype=torch.int32)
    for i in range(n):
        omega += (r_and_s >> i) & 1
        
    phase_lookup = torch.tensor([1.0, -1j, -1.0, 1j], dtype=torch.complex128, device=device)
    phases = phase_lookup[omega % 4] / N
    
    c_p = A_fwht * phases
    c_p_flat = c_p.flatten()
    
    # 4. High-Performance Filtering 
    mags_sq = torch.abs(c_p_flat)**2
    sorted_mags, sorted_idx = torch.sort(mags_sq, descending=True)
    
    if k is not None:
        selected_idx = sorted_idx[:k]
    elif eta is not None:
        csum = torch.cumsum(sorted_mags, dim=0)
        target = eta * csum[-1]
        
        # Determine exact threshold cutoff point utilizing native binary search tree equivalent
        num_elements = torch.searchsorted(csum, target, right=False).item() + 1
        num_elements = min(num_elements, len(csum))
        selected_idx = sorted_idx[:num_elements]
    else:
        selected_idx = sorted_idx
        
    # Execute CPU offloading just for building final strings logic
    selected_idx_cpu = selected_idx.cpu().tolist()
    c_p_flat_cpu = c_p_flat.cpu().tolist()
    
    results = []
    for idx in selected_idx_cpu:
        r_val = idx // N
        s_val = idx % N
        coef = c_p_flat_cpu[idx]
        string = pauli_index_to_string(r_val, s_val, n)
        results.append((string, coef))
        
    return results

def ExtractGoldenStrings(Delta_hat, n: int, eta: float = None, k: int = None):
    """
    Extracts the most significant Pauli strings using Fast Walsh-Hadamard Transform 
    based decomposition exactly achieving O(N^2 log N) limits.
    
    Returns a list of tuples (PauliString, Coefficient) sorted by descending magnitude.
    
    Args:
        Delta_hat: Dense complex matrix of shape (2^n, 2^n). (NumPy array or PyTorch tensor)
        n: Number of qubits (matrix matches 2^n x 2^n dimensions).
        eta: Target energy fractional threshold in (0.0, 1.0]. Extracts strings whose 
             combined coefficient squared magnitude satisfies `eta * ||Delta||_F^2 / 2^n`.
        k: Return top k dominant Pauli strings tightly.
        
    Returns:
        List[Tuple[str, complex]]
    """
    if torch is not None and isinstance(Delta_hat, torch.Tensor):
        return _extract_golden_torch(Delta_hat, n, eta, k)
    elif isinstance(Delta_hat, np.ndarray):
        return _extract_golden_numpy(Delta_hat, n, eta, k)
    else:
        raise TypeError("Delta_hat must be a NumPy array or a PyTorch tensor.")
