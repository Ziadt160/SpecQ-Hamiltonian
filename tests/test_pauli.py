"""Pauli algebra and the FWHT decomposition."""
import numpy as np
import pytest

from src.generators.spectral_pauli_generator import (
    count_dead_strings,
    is_dead_string,
    pauli_coefficients,
)
from src.utils.pauli_decompose import ExtractGoldenStrings, extract_golden_strings
from src.utils.pauli_utils import (
    classify_pauli_string,
    generate_pauli_strings,
    get_pauli_tensor,
)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_basis_size_and_weight_filter(n):
    strings = generate_pauli_strings(n)
    assert len(strings) == 4 ** n == len(set(strings))
    low = generate_pauli_strings(n, max_weight=1)
    assert len(low) == 1 + 3 * n
    assert all(sum(c != 'I' for c in s) <= 1 for s in low)


def test_pauli_tensors_are_hermitian_unitary_and_orthogonal():
    strings = generate_pauli_strings(2)
    mats = [get_pauli_tensor(s) for s in strings]
    for P in mats:
        assert np.allclose(P, P.conj().T)
        assert np.allclose(P @ P, np.eye(4))
    gram = np.array([[np.trace(A.conj().T @ B) for B in mats] for A in mats])
    assert np.allclose(gram, 4 * np.eye(16))


def test_classify_pauli_string():
    assert classify_pauli_string('II') == 'C0'
    assert classify_pauli_string('IZ') == 'C1Z'
    assert classify_pauli_string('XI') == 'C1X'
    assert classify_pauli_string('ZX') == 'C2ZX'
    assert classify_pauli_string('XX') == 'C2XX'
    assert classify_pauli_string('YI') == 'CY'
    assert classify_pauli_string('YY') == 'Other'


@pytest.mark.parametrize("n", [2, 3, 4])
def test_dead_strings_are_identically_zero_on_real_inputs(n):
    """README section 3.2: odd-Y strings return exactly zero for real x."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 2 ** n))
    strings = generate_pauli_strings(n)
    dead = [s for s in strings if is_dead_string(s)]
    assert len(dead) == count_dead_strings(n) == (4 ** n - 2 ** n) // 2
    for s in strings:
        vals = np.einsum('bi,ij,bj->b', X, get_pauli_tensor(s), X)
        if is_dead_string(s):
            assert np.max(np.abs(vals)) < 1e-12
        else:
            assert np.max(np.abs(vals)) > 1e-6


@pytest.mark.parametrize("n", [1, 2, 3])
def test_fwht_matches_naive_trace_and_reconstructs(n):
    rng = np.random.default_rng(42)
    N = 2 ** n
    Delta = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    paulis = {s: get_pauli_tensor(s) for s in generate_pauli_strings(n)}

    reconstructed = np.zeros((N, N), dtype=complex)
    for s, coef in extract_golden_strings(Delta, n):
        assert np.isclose(coef, np.trace(paulis[s] @ Delta) / N, atol=1e-10), s
        reconstructed += coef * paulis[s]
    assert np.allclose(reconstructed, Delta, atol=1e-10)


def test_fwht_agrees_with_dense_projection_on_symmetric_delta():
    """The two selection back-ends must assign every string the same coefficient."""
    n = 3
    rng = np.random.default_rng(7)
    A = rng.standard_normal((2 ** n, 2 ** n))
    Delta = A + A.T
    strings = generate_pauli_strings(n)
    dense = dict(zip(strings, pauli_coefficients(Delta, strings)))
    for s, coef in extract_golden_strings(Delta, n):
        assert np.isclose(coef, dense[s], atol=1e-12), s


def test_fwht_top_k_and_eta():
    Delta = np.eye(4, dtype=complex)
    golden = extract_golden_strings(Delta, 2, k=1)
    assert golden == [('II', pytest.approx(1.0))]
    assert [s for s, _ in extract_golden_strings(Delta, 2, eta=0.99)] == ['II']


def test_fwht_torch_parity_and_legacy_alias():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    Delta = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    golden_np = extract_golden_strings(Delta, 2)
    golden_pt = ExtractGoldenStrings(torch.tensor(Delta, dtype=torch.complex128), 2)
    for (s_np, c_np), (s_pt, c_pt) in zip(golden_np, golden_pt):
        assert s_np == s_pt
        assert np.isclose(c_np, c_pt, atol=1e-10)
