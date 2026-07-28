import numpy as np
import time
from src.utils.pauli_decompose import ExtractGoldenStrings

def generate_all_paulis(n):
    """
    Generate the complete 4^n set of Pauli strings and their matrices 
    ordered MSB (left) to LSB (right).
    """
    import itertools
    mats = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }
    
    results = []
    for p_tuple in itertools.product('IXYZ', repeat=n):
        s = "".join(p_tuple)
        mat = np.array([[1]], dtype=complex)
        for char in s:
            mat = np.kron(mat, mats[char])
        results.append((s, mat))
    return results

def naive_pauli_decomposition(Delta, n):
    """
    Mathematically exact standard inner product O(4^n) implementation.
    """
    paulis = generate_all_paulis(n)
    coeffs = {}
    for s, P in paulis:
        # Note: P = P.conj().T (Hermitian)
        c = np.trace(P @ Delta) / (2**n)
        coeffs[s] = c
    return coeffs

def test_correctness_numpy(n):
    """
    Stress-test random real & complex matrices comparing exact Hilbert sums natively.
    """
    np.random.seed(42)
    N = 2**n
    
    Delta = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    
    # Run O(N^2 logN)
    golden_strings = ExtractGoldenStrings(Delta, n)
    
    # Run standard trace product
    naive_coeffs = naive_pauli_decomposition(Delta, n)
    
    reconstructed = np.zeros((N, N), dtype=complex)
    paulis = dict(generate_all_paulis(n))
    
    for string, coef in golden_strings:
        expected_coef = naive_coeffs[string]
        # Assert each isolated matrix aligns computationally with traditional traces
        assert np.isclose(coef, expected_coef, atol=1e-10), f"Mismatch for {string}"
        reconstructed += coef * paulis[string]
        
    assert np.allclose(reconstructed, Delta, atol=1e-10), "Full matrix rebuilding step invalid!"

def test_pytorch_parity():
    """ Verify computational parity between tensor architectures. """
    try:
        import torch
    except ImportError:
        print("PyTorch not installed, skipping.")
        return
        
    n = 2
    N = 2**n

    
    np.random.seed(42)
    Delta_np = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    Delta_pt = torch.tensor(Delta_np, dtype=torch.complex128)
    
    golden_np = ExtractGoldenStrings(Delta_np, n)
    golden_pt = ExtractGoldenStrings(Delta_pt, n)
    
    for (s_np, c_np), (s_pt, c_pt) in zip(golden_np, golden_pt):
        assert s_np == s_pt
        assert np.isclose(c_np, c_pt, atol=1e-10)

def test_top_k():
    n = 2
    N = 2**n
    Delta = np.eye(N, dtype=complex) 
    
    golden = ExtractGoldenStrings(Delta, n, k=1)
    
    assert len(golden) == 1
    assert golden[0][0] == 'II'
    assert np.isclose(golden[0][1], 1.0)
    
def test_eta_selection():
    n = 2
    N = 2**n
    Delta = np.eye(N, dtype=complex) 
    
    # Identity matrix projects completely to the 'II' string having weight 1.0.
    golden = ExtractGoldenStrings(Delta, n, eta=0.99)
    assert len(golden) == 1
    assert golden[0][0] == 'II'

if __name__ == "__main__":
    
    # Manually execute tests
    print("Testing structural correctness (n=2)...")
    test_correctness_numpy(2)
    test_top_k()
    
    print("Running Benchmark...")
    n = 8
    N = 2**n
    np.random.seed(111)
    Delta = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    
    start = time.time()
    extracted = ExtractGoldenStrings(Delta, n)
    end = time.time()
    
    print(f"NumPy FWHT extraction n={n} ({N}x{N}) took: {end-start:.4f} secs.")
    
    try:
        import torch
        if torch.cuda.is_available():
            Delta_pt = torch.tensor(Delta, dtype=torch.complex128).cuda()
            
            # WARMUP
            _ = ExtractGoldenStrings(Delta_pt, n)
            torch.cuda.synchronize()
            
            start = time.time()
            extracted_pt = ExtractGoldenStrings(Delta_pt, n)
            torch.cuda.synchronize()
            end = time.time()
            print(f"PyTorch CUDA FWHT n={n} ({N}x{N}) took: {end-start:.4f} secs.")
        else:
             print("Skipping PyTorch CUDA benchmark; No GPU detected.")
    except Exception as e:
        print("PyTorch error:", str(e))
