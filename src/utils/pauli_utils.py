import numpy as np
from itertools import product
from functools import reduce

def get_pauli_matrix(label):
    """
    Returns the 2x2 Pauli matrix for the given label.
    
    Args:
        label (str): 'I', 'X', 'Y', or 'Z'.
        
    Returns:
        np.array: 2x2 complex numpy array.
    """
    if label == 'I':
        return np.eye(2, dtype=complex)
    elif label == 'X':
        return np.array([[0, 1], [1, 0]], dtype=complex)
    elif label == 'Y':
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif label == 'Z':
        return np.array([[1, 0], [0, -1]], dtype=complex)
    else:
        raise ValueError(f"Invalid Pauli label: {label}")

def generate_pauli_strings(n_qubits, max_weight=None):
    """
    Generates all Pauli strings of length n_qubits.
    
    Args:
        n_qubits (int): Number of qubits.
        max_weight (int, optional): Maximum Hamming weight (non-Identity terms).
        
    Returns:
        list of str: List of Pauli strings (e.g., ['II', 'IX', ...]).
    """
    labels = ['I', 'X', 'Y', 'Z']
    all_strings = [''.join(p) for p in product(labels, repeat=n_qubits)]
    
    if max_weight is not None:
        filtered = []
        for s in all_strings:
            weight = sum(1 for c in s if c != 'I')
            if weight <= max_weight:
                filtered.append(s)
        return filtered
    return all_strings

def classify_pauli_string(s):
    """
    Classifies a Pauli string into one of the defined experimental classes.
    
    Classes:
    - C0: Identity only ('II...I')
    - C1Z: Exactly one Z, rest I ('IZI')
    - C1X: Exactly one X, rest I ('IXI')
    - C2ZX: Exactly one Z and one X ('IZX', 'XZI')
    - C2XX: Exactly two X's ('IXX')
    - CY: Any string with an odd number of Y's (should vanish for real reals)
    - Other: Anything else
    
    Args:
        s (str): Pauli string.
        
    Returns:
        str: Class label.
    """
    counts = {c: s.count(c) for c in ['I', 'X', 'Y', 'Z']}
    
    # Check for Y first (odd number of Ys -> CY)
    if counts['Y'] % 2 != 0:
        return 'CY'
    
    # Check C0
    if counts['X'] == 0 and counts['Z'] == 0 and counts['Y'] == 0:
        return 'C0'
        
    # Check C1Z
    if counts['Z'] == 1 and counts['X'] == 0 and counts['Y'] == 0:
        return 'C1Z'
        
    # Check C1X
    if counts['X'] == 1 and counts['Z'] == 0 and counts['Y'] == 0:
        return 'C1X'
        
    # Check C2ZX
    if counts['Z'] == 1 and counts['X'] == 1 and counts['Y'] == 0:
        return 'C2ZX'
        
    # Check C2XX
    if counts['X'] == 2 and counts['Z'] == 0 and counts['Y'] == 0:
        return 'C2XX'
        
    return 'Other'

def get_pauli_tensor(s):
    """
    Computes the full tensor product matrix for a Pauli string.
    
    Args:
        s (str): Pauli string.
        
    Returns:
        np.array: 2^N x 2^N matrix.
    """
    matrices = [get_pauli_matrix(c) for c in s]
    return reduce(np.kron, matrices)
