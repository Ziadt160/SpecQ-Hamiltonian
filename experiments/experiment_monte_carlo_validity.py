import numpy as np
import matplotlib.pyplot as plt
from spectral_pauli_generator import generate_spectral_pauli_strings
from analysis_canonical_patterns import load_20newsgroups_projected
from pauli_utils import get_pauli_tensor, generate_pauli_strings
from sklearn.model_selection import train_test_split

def compute_mc_estimate(X, y, pauli_string, n_samples=1000):
    """
    Estimates c_P using Monte Carlo sampling.
    c_P = Tr(Delta * P) / 2^n
        = Tr((R1 - R0) * P) / 2^n
        = (Tr(R1 P) - Tr(R0 P)) / 2^n
        
    Tr(R_y * P) = Tr( E[x x^T] * P ) = E[ Tr(x x^T P) ] = E[ x^T P x ]
    
    So c_P = ( E[x^T P x | 1] - E[x^T P x | 0] ) / 2^n (?)
    Wait. 
    Delta = R1 - R0. 
    Tr(Delta P) = Tr(R1 P) - Tr(R0 P).
    R1 = sum(x1 x1^T) / N1
    Tr(R1 P) = Tr( sum(x1 x1^T) P / N1 )
             = sum( Tr(x1 x1^T P) ) / N1
             = sum( x1^T P x1 ) / N1
             = Mean( x^T P x ) over class 1.
             
    So yes! c_P = ( Mean_1(x^T P x) - Mean_0(x^T P x) ) / 2^n.
    
    NOTE: The 1/2^n factor is a convention of the inner product. 
    If we stick to our implementation:
    Our code does `np.trace(Delta @ P) * factor`.
    So the ESTIMATOR is just:
    ( Mean(x1^T P x1) - Mean(x0^T P x0) ) / 2^n.
    
    This is O(M) samples. No matrix construction needed.
    """
    
    # 1. Sample Subset
    # X and y are full dataset.
    
    X0 = X[y == 0]
    X1 = X[y == 1]
    
    # Subsample if requested (though for this validaton we want to see convergence)
    # Actually, let's use all samples but pretend we didn't construct the matrix.
    # Or subsample to characterize error vs samples.
    
    idx0 = np.random.choice(len(X0), min(len(X0), n_samples), replace=True)
    idx1 = np.random.choice(len(X1), min(len(X1), n_samples), replace=True)
    
    batch0 = X0[idx0]
    batch1 = X1[idx1]
    
    # Compute quadratic forms x^T P x
    # P matrix
    P_tensor = get_pauli_tensor(pauli_string)
    # Matrix form? 
    # get_pauli_tensor returns numpy array now.
    P_matrix = P_tensor
    
    # Batch compute
    # term0 = diag(batch0 @ P @ batch0.T)
    # Efficient: sum( (X @ P) * X, axis=1 )
    
    term0 = np.sum((batch0 @ P_matrix) * batch0, axis=1) # (M,)
    term1 = np.sum((batch1 @ P_matrix) * batch1, axis=1) # (M,)
    
    mean0 = np.mean(term0)
    mean1 = np.mean(term1)
    
    dim = X.shape[1] # 2^n
    
    # c_P estimate
    c_hat = (mean1 - mean0) / dim
    
    return c_hat

def run_mc_validation():
    print("Loading Data (N=4)...")
    X, y = load_20newsgroups_projected(n_qubits=4)
    # No split needed, just validating calculation
    
    # 1. Compute EXACT coefficients (Ground Truth)
    print("Computing Exact Spectral Coefficients...")
    # This uses the full matrix method inside generate_spectral_pauli_strings
    _, exact_coefs, _ = generate_spectral_pauli_strings(X, y, 4) 
    # Wait, generate_spectral_pauli_strings returns: sorted_strings, coefficients, magnitudes
    # And it sorts them.
    # To compare, we need the SAME strings.
    # So let's extract unsorted? No, generate_spectral sorts.
    
    # Let's just generate strings and compute exact delta manually to be sure about ordering.
    
    strings = generate_pauli_strings(4)
    
    # Re-compute Exact Delta manually for certainty
    dim = 16
    X0 = X[y==0]
    X1 = X[y==1]
    R0 = (X0.T @ X0) / len(X0)
    R1 = (X1.T @ X1) / len(X1)
    Delta = R1 - R0
    
    exact_values = []
    print(f"Computing exact values for {len(strings)} strings...")
    for s in strings:
        P = get_pauli_tensor(s)
        val = np.trace(Delta @ P) / dim
        exact_values.append(val)
        
    exact_values = np.array(exact_values)
    
    # 2. Compute Monte Carlo Estimate (using limited samples)
    print("Computing Monte Carlo Estimates (N=100 samples)...")
    mc_values_100 = []
    
    for s in strings:
        c_hat = compute_mc_estimate(X, y, s, n_samples=100)
        mc_values_100.append(c_hat)
        
    mc_values_100 = np.array(mc_values_100)
    
    # 3. Compare
    # Correlation
    corr = np.corrcoef(np.abs(exact_values), np.abs(mc_values_100))[0, 1]
    
    print(f"\nCorrelation (Exact vs MC-100): {corr:.4f}")
    
    # Sort ordering match
    # Top-32 overlap
    exact_indices = np.argsort(np.abs(exact_values))[::-1][:32]
    mc_indices = np.argsort(np.abs(mc_values_100))[::-1][:32]
    
    overlap = len(set(exact_indices) & set(mc_indices))
    print(f"Top-32 Overlap: {overlap}/32 ({overlap/32:.1%})")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(np.abs(exact_values), np.abs(mc_values_100), alpha=0.5, s=10)
    plt.plot([0, max(np.abs(exact_values))], [0, max(np.abs(exact_values))], 'r--')
    plt.xlabel('Exact |c_P|')
    plt.ylabel('MC Estimated |c_P| (N=100)')
    plt.title(f'Monte Carlo Validation (R={corr:.3f})')
    
    plt.subplot(1, 2, 2)
    plt.hist(np.abs(mc_values_100 - exact_values), bins=30)
    plt.xlabel('Estimation Error')
    plt.title('Error Distribution')
    
    plt.tight_layout()
    plt.savefig('../results/mc_validation.png')
    print("Validation plot saved to results/mc_validation.png")

if __name__ == "__main__":
    run_mc_validation()
