"""SIM / HAM / PEFF models and their noisy and self-selecting variants."""
import numpy as np
import pytest
import torch

from src.models.exact_sim_classifier import ExactSIMClassifier
from src.models.fwht_sim_classifier import FWHTExactSIMClassifier, FWHTSIMClassifier
from src.models.hamiltonian_variants import HAMClassifier, PEFFClassifier
from src.models.nisq_sim_classifier import NISQSIMClassifier, thermal_parameters
from src.models.sim_classifier import SIMClassifier
from src.models.spectral_sim_classifier import SpectralExactSIMClassifier, SpectralSIMClassifier
from src.utils.data_generator import generate_dataset
from src.utils.pauli_utils import generate_pauli_strings

BASIS_2Q = ["II", "IZ", "ZI", "ZZ"]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# ExactSIMClassifier (Eq. 9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_classes, out_shape", [(1, (4,)), (3, (4, 3))])
def test_exact_sim_shapes_for_flat_and_sequence_inputs(num_classes, out_shape):
    model = ExactSIMClassifier(n_qubits=2, num_classes=num_classes, pauli_strings=BASIS_2Q)
    assert model(torch.randn(4, 4, dtype=torch.float64)).shape == out_shape
    assert model(torch.randn(4, 3, 4, dtype=torch.float64)).shape == out_shape


def test_exact_sim_alpha_normalisation_is_a_reparametrisation():
    """The 1/2^n prefactor is absorbed exactly by rescaling w."""
    torch.manual_seed(0)
    a = ExactSIMClassifier(n_qubits=2, pauli_strings=BASIS_2Q, normalize_alpha=False)
    b = ExactSIMClassifier(n_qubits=2, pauli_strings=BASIS_2Q, normalize_alpha=True)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        b.w.mul_(4)
    x = torch.randn(5, 4, dtype=torch.float64)
    assert torch.allclose(a(x), b(x))


def test_nisq_with_noise_off_reproduces_exact_sim():
    """CHANGELOG, "Noise model": zero noise must equal the noiseless model."""
    strings = generate_pauli_strings(3)
    torch.manual_seed(0)
    exact = ExactSIMClassifier(n_qubits=3, n_layers=3, pauli_strings=strings)
    nisq = NISQSIMClassifier(n_qubits=3, n_layers=3, pauli_strings=strings,
                             t1=np.inf, t2=np.inf, p_gate_1q=0.0, p_gate_2q=0.0,
                             p_readout=0.0)
    nisq.load_state_dict(exact.state_dict(), strict=False)
    e_exact = exact.pauli_expectations().detach().numpy()
    e_nisq = nisq.pauli_expectations().detach().numpy()
    assert np.max(np.abs(e_exact - e_nisq)) < 1e-10


def test_thermal_parameters():
    assert thermal_parameters(np.inf, np.inf, 1e-7) == (0.0, 0.0)
    gamma_amp, gamma_phase = thermal_parameters(100e-6, 80e-6, 200e-9)
    assert 0 < gamma_amp < 1e-2 and 0 < gamma_phase < 1e-2
    with pytest.raises(ValueError):
        thermal_parameters(50e-6, 120e-6, 200e-9)          # T2 > 2 T1


def test_readout_error_attenuates_by_weight():
    nisq = NISQSIMClassifier(n_qubits=2, n_layers=1, pauli_strings=["II", "XI", "XX"],
                             p_readout=0.1)
    assert torch.allclose(nisq.readout_attenuation,
                          torch.tensor([1.0, 0.8, 0.64], dtype=torch.float64))


# ---------------------------------------------------------------------------
# HAM / PEFF (Eq. 2-3, Sec. 3.2)
# ---------------------------------------------------------------------------

def test_ham_and_peff_parameter_scaling():
    """README section 5.6: at n=4, L=3 HAM has 292 parameters and PEFF 54."""
    ham, peff = HAMClassifier(n_qubits=4, n_layers=3), PEFFClassifier(n_qubits=4, n_layers=3)
    assert n_params(ham) == 16 * 16 + 3 * 4 * 3
    assert n_params(peff) == 16 + 1 + 1 + 3 * 4 * 3
    x = torch.randn(6, 16, dtype=torch.float64)
    for model in (ham, peff):
        out = model(x)
        assert out.shape == (6,) and torch.all((out > 0) & (out < 1))


# ---------------------------------------------------------------------------
# Classical SIM and the self-selecting wrappers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime, basis_class", [
    ("linear", "Z"), ("pairwise", "XX"), ("conditional", "ZX"),
])
def test_sim_solves_each_synthetic_regime_with_its_pauli_class(regime, basis_class):
    """CHANGELOG, "Synthetic data": each regime is solved by its own class."""
    X_tr, y_tr = generate_dataset(500, 4, regime, random_state=42)
    X_te, y_te = generate_dataset(200, 4, regime, random_state=99)
    assert set(np.unique(y_tr)) == {0, 1}
    basis = [s for s in generate_pauli_strings(4)
             if 'Y' not in s and ''.join(sorted(s.replace('I', ''))) == ''.join(sorted(basis_class))]
    model = SIMClassifier(pauli_strings=basis, C=100.0, random_state=42).fit(X_tr, y_tr)
    assert model.score(X_te, y_te) > 0.95


def test_sim_rejects_mismatched_dimension():
    with pytest.raises(ValueError):
        SIMClassifier(pauli_strings=["ZZ"]).fit(np.ones((4, 8)), np.array([0, 1, 0, 1]))


@pytest.mark.parametrize("cls", [SpectralSIMClassifier, FWHTSIMClassifier])
def test_self_selecting_classical_sim(cls):
    X_tr, y_tr = generate_dataset(400, 3, "pairwise", random_state=0)
    X_te, y_te = generate_dataset(200, 3, "pairwise", random_state=1)
    model = cls(n_qubits=3, top_k=12, C=10.0).fit(X_tr, y_tr)
    assert len(model.pauli_strings) == 12
    assert np.isrealobj(np.asarray(model.spectral_coefs_))
    assert model.score(X_te, y_te) > 0.8


@pytest.mark.parametrize("cls", [SpectralExactSIMClassifier, FWHTExactSIMClassifier])
def test_refit_reselects_basis(cls):
    """A second fit() used to silently keep the first fit's Pauli basis."""
    X_a, y_a = generate_dataset(120, 2, "linear", random_state=0)
    X_b, y_b = generate_dataset(120, 2, "pairwise", random_state=0)
    model = cls(n_qubits=2, n_layers=1, top_k=2)
    first = list(model.fit(X_a, y_a, epochs=1, verbose=False).pauli_strings_)
    second = list(model.fit(X_b, y_b, epochs=1, verbose=False).pauli_strings_)
    assert first != second
    proba = model.predict_proba(X_b)
    assert proba.shape == (120,) and np.all((proba > 0) & (proba < 1))
    assert set(np.unique(model.predict(X_b))) <= {0, 1}


def test_fixed_basis_is_respected():
    X, y = generate_dataset(60, 2, "linear", random_state=0)
    model = SpectralExactSIMClassifier(n_qubits=2, n_layers=1, pauli_strings=["ZI", "IZ"])
    model.fit(X, y, epochs=1, verbose=False)
    assert model.pauli_strings_ == ["ZI", "IZ"]
