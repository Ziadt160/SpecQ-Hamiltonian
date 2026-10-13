"""Spectral, FWHT and QMI basis selection."""
import numpy as np
import pytest

from src.generators.qmi_pauli_generator import qmi_score
from src.generators.spectral_pauli_generator import (
    compute_delta,
    generate_spectral_pauli_strings,
    get_adaptive_spectral_paulis,
    is_dead_string,
)
from src.models.fwht_sim_classifier import generate_fwht_pauli_strings


@pytest.fixture
def shifted_classes():
    """Two classes with different means and different covariances."""
    rng = np.random.default_rng(0)
    X0 = rng.standard_normal((200, 8)) + 0.5
    X1 = rng.standard_normal((200, 8)) * np.linspace(0.5, 2.0, 8) - 0.5
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    return X, y


def test_delta_definitions(shifted_classes):
    X, y = shifted_classes
    cov = compute_delta(X, y, moment='covariance')
    second = compute_delta(X, y, moment='second_moment')
    assert np.allclose(cov, np.cov(X[y == 1], rowvar=False) - np.cov(X[y == 0], rowvar=False))
    assert np.allclose(cov, cov.T)
    # Non-zero class means: the two definitions must differ (CHANGELOG, "Selection").
    assert not np.allclose(cov, second)
    with pytest.raises(ValueError):
        compute_delta(X, y, moment='variance')


def test_degenerate_class_returns_zero_delta(shifted_classes):
    X, _ = shifted_classes
    with pytest.warns(RuntimeWarning):
        Delta = compute_delta(X, np.zeros(len(X), int))
    assert not Delta.any()
    with pytest.warns(RuntimeWarning):
        assert get_adaptive_spectral_paulis(X, np.zeros(len(X), int), 3) == ([], [], 0)


def test_dead_strings_rank_below_every_live_string(shifted_classes):
    """README section 5.2: any spectral top-k is 100% live."""
    X, y = shifted_classes
    strings, _, mags = generate_spectral_pauli_strings(X, y, 3)
    first_dead = next(i for i, s in enumerate(strings) if is_dead_string(s))
    assert first_dead == (4 ** 3 + 2 ** 3) // 2      # 36 live strings at n=3
    assert all(is_dead_string(s) for s in strings[first_dead:])
    assert max(mags[first_dead:]) < 1e-12
    assert list(mags) == sorted(mags, reverse=True)


def test_top_k_and_adaptive_cutoff(shifted_classes):
    X, y = shifted_classes
    strings, coefs = generate_spectral_pauli_strings(X, y, 3, top_k=5)
    assert len(strings) == len(coefs) == 5

    all_strings, _, mags = generate_spectral_pauli_strings(X, y, 3)
    sel, _, k = get_adaptive_spectral_paulis(X, y, 3, eta=0.9)
    share = np.cumsum(mags) / np.sum(mags)
    assert sel == all_strings[:k]
    assert share[k - 1] >= 0.9 and (k == 1 or share[k - 2] < 0.9)


def test_fwht_selection_matches_spectral_selection(shifted_classes):
    X, y = shifted_classes
    for moment in ('covariance', 'second_moment'):
        spec, spec_coefs = generate_spectral_pauli_strings(X, y, 3, top_k=10, moment=moment)
        fwht, fwht_coefs = generate_fwht_pauli_strings(X, y, 3, top_k=10, moment=moment)
        assert set(spec) == set(fwht)
        assert np.allclose(sorted(np.abs(spec_coefs)), sorted(np.abs(fwht_coefs)))


def test_qmi_orders_by_dependence():
    """CHANGELOG: the old scorer ranked a perfect predictor last."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    noise = rng.standard_normal(300)
    perfect = y + 0.01 * noise
    useful = y + 0.8 * noise
    independent = rng.standard_normal(300)
    scores = [qmi_score(f, y) for f in (perfect, useful, independent)]
    assert scores[0] > scores[1] > scores[2] >= 0.0
    assert qmi_score(np.ones(300), y) == 0.0
