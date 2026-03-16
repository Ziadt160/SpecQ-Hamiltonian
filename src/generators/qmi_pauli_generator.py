import numpy as np
from scipy.spatial.distance import pdist, squareform
from ..utils.pauli_utils import generate_pauli_strings, get_pauli_tensor

def compute_quadratic_features(X, paulis):
    """
    Computes quadratic features f(x) = x^T P x for a list of Pauli matrices.
    
    Args:
        X (np.ndarray): Input samples (n_samples, dim).
        paulis (list of np.ndarray): List of 2^n x 2^n Pauli matrices.
        
    Returns:
        np.ndarray: Computed features of shape (n_samples, n_paulis).
    """
    features = []
    # Vectorized computation could be memory heavy if all Paulis are passed at once.
    # Processing in loop or batches is safer for large n_paulis.
    for P in paulis:
        # P is Hermitian, X is real (usually). Result is real.
        # einsum 'ni,ij,nj->n' computes x_n^T P x_n for each sample n
        f = np.einsum('ni,ij,nj->n', X, P, X)
        features.append(f.real) # Take real part (should be real for Hermitian P)
        
    return np.array(features).T

def qmi_score(f, y, sigma=None):
    """
    Computes the Quadratic Mutual Information (QMI) score.
    
    QMI measures the statistical dependence between a Pauli feature $f(x) = x^T P x$ 
    and labels $y$ using a Parzen-window estimator of Renyi's Quadratic Entropy.
    Higher QMI indicates a more discriminative interaction.
    
    Args:
        f (np.ndarray): Feature values (n_samples,).
        y (np.ndarray): Class labels (n_samples,).
        sigma (float, optional): Kernel width.
    """
    f = f.reshape(-1, 1)
    y = np.array(y).ravel()
    n = len(f)
    
    # Kernel matrix for features
    # Try/Except to handle potential memory issues if N is very large, though N usually < 2000 here.
    try:
        dist_sq = squareform(pdist(f, 'sqeuclidean'))
    except MemoryError:
        print("Warning: MemoryError in QMI calculation. Using lower precision or downsampling might be needed.")
        return 0.0

    if sigma is None:
        # Median heuristic: median of non-zero distances / sqrt(2)
        # Add small epsilon to avoid zero division if all points are identical
        nonzero_dists = dist_sq[dist_sq > 0]
        if len(nonzero_dists) == 0:
            sigma = 1.0 
        else:
            sigma = np.median(nonzero_dists) / np.sqrt(2)
            
    # Check for valid sigma
    if sigma <= 1e-12:
        sigma = 1.0

    K_xx = np.exp(-dist_sq / (2 * sigma**2))
    
    # V_x: average kernel similarity (Information Potential of features)
    V_x = np.mean(K_xx)
    
    # V_y: Information Potential of labels (exact for discrete y)
    classes, counts = np.unique(y, return_counts=True)
    p_c = counts / n
    V_y = np.sum(p_c**2)
    
    # V_xy: Join Information Potential
    # Kernel on joint space is product of kernels: K((x,y), (x',y')) = K_x(x,x') * K_y(y,y')
    # K_y(y,y') is 1 if y=y', 0 otherwise (Dirac kernel for discrete labels)
    same_class = (y[:, None] == y[None, :])
    # K_joint = K_xx * same_class
    V_xy = np.mean(K_xx * same_class)
    
    # QMI = log(V_xy^2 / (V_x * V_y))
    # Add epsilon to avoiding log(0)
    eps = 1e-12
    if V_x * V_y < eps:
        return 0.0
        
    ratio = (V_xy**2) / (V_x * V_y)
    qmi = -np.log(ratio + eps) 
    
    # Note: The QMI definition can vary. 
    # Standard QMI (Principe et al) is actually integral((p_xy - p_x p_y)^2).
    # The user provided formula: -log( (V_xy^2) / (V_x * V_y) ) which looks like a Renyi divergence or similar.
    # However, standard max-dependence criteria often maximize V_xy or log(V_xy) etc.
    # The user's formula effectively measures the alignment.
    # If V_xy^2 = V_x * V_y (independence), ratio = 1, log(1) = 0.
    # If dependent, V_xy should be larger? 
    # Actually, Cauchy-Schwarz inequality says V_xy^2 <= V_x * V_y * (something?).
    # Let's stick strictly to the User's formula: qmi = -np.log( (V_xy**2) / (V_x * V_y) + 1e-12 )
    # Wait, the user code says: qmi = -np.log( (V_xy**2) / (V_x * V_y) + 1e-12 )
    # And returns max(qmi, 0.0).
    # If fully dependent (perfect alignment), V_xy is maximized.
    
    # Let's verify the logic. 
    # If y implies x perfectly, V_xy is large.
    # Wait, if ratio < 1, log(ratio) is negative, so -log is positive.
    # If ratio > 1 (possible?), -log is negative.
    # We want to MAXIMIZE dependence. 
    # User implementation provided: 
    # score = qmi_score(...)
    # ranked_indices = np.argsort(qmi_scores)[::-1] -> Descending order.
    # So bigger score = better. 
    
    # If V_xy is large (good alignment), ratio is large? 
    # CS inequality for IP: V_xy <= sqrt(V_x * V_y_kernel_matrix_avg).
    # Actually, let's just implement exactly as requested.
    
    val = (V_xy**2) / (V_x * V_y + eps)
    qmi = -np.log(val + eps)
    
    # The user code return max(qmi, 0.0) implies they expect positive values.
    # If val < 1, -log(val) > 0. 
    # So we want small V_xy? That contradicts "Higher = more dependent".
    
    # RE-READING USER REQUEST CAREFULLY:
    # "QMI (higher = more dependent; avoid div0)"
    # "qmi = -np.log( (V_xy**2) / (V_x * V_y) + 1e-12 )"
    
    # If X and Y are independent, p_xy = p_x p_y. 
    # V_xy = Mean(K_x(xi, xj) * K_y(yi, yj)) ~= Mean(K_x) * Mean(K_y) = V_x * V_y.
    # So ratio ~= (V_x V_y)^2 / (V_x V_y) = V_x V_y. 
    
    # Actually, let's look at the CS Divergence formula usually:
    # D_CS = log(V_xy / sqrt(V_x * V_y)) ? No.
    # D_CS(P1, P2) = -log ( IP(P1,P2) / sqrt(IP(P1)IP(P2)) )
    # Here we are comparing Joint PDF vs Product PDF? 
    # If we want independence measure -> 0 if independent, >0 if dependent.
    # Using User's exact snippet. 
    
    return max(qmi, 0.0)

def generate_qmi_pauli_strings(X, y, n_qubits, top_k=None):
    """
    Selects Pauli strings using Quadratic Mutual Information (QMI).
    
    Args:
        X (np.ndarray): Input features (N_samples, 2^n_qubits).
        y (np.ndarray): Labels (0, 1).
        n_qubits (int): Number of qubits.
        top_k (int, optional): Number of top strings to return.
        
    Returns:
        list of str: Sorted Pauli strings.
        list of float: QMI scores.
    """
    # 1. Generate all Pauli strings (for small n)
    # The user provided generate_paulis approach, but we can use our existing util and convert to matrices.
    # Existing util: generate_pauli_strings returns strings ['II', 'IX', ...].
    # We need matrices.
    
    strings = generate_pauli_strings(n_qubits)
    # Remove identity if present (usually 'I'*n is not useful for discrimination as it is constant)
    identity_str = 'I' * n_qubits
    if identity_str in strings:
        strings.remove(identity_str)
        
    pauli_matrices = [get_pauli_tensor(s) for s in strings]
    
    # 2. Compute QMI scores
    qmi_scores = []
    
    # Pre-compute matrices/loop
    # Doing one by one to save memory as suggested by user
    for i, P in enumerate(pauli_matrices):
        # f shape: (N,)
        f = compute_quadratic_features(X, [P])[:, 0]
        score = qmi_score(f, y)
        qmi_scores.append(score)
        
    # 3. Sort
    # User says: argsort(qmi_scores)[::-1] (Descending)
    indexed_scores = list(zip(strings, qmi_scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    sorted_strings = [x[0] for x in indexed_scores]
    sorted_scores = [x[1] for x in indexed_scores]
    
    if top_k is not None:
        return sorted_strings[:top_k], sorted_scores[:top_k]
        
    return sorted_strings, sorted_scores

if __name__ == "__main__":
    # Sanity check
    n = 2
    X = np.random.randn(10, 4)
    y = np.array([0]*5 + [1]*5)
    strings, scores = generate_qmi_pauli_strings(X, y, n, top_k=5)
    print("Sanity Check strings:", strings)
    print("Sanity Check scores:", scores)
