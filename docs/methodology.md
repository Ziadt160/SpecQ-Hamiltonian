# Methodology

Implementation notes that complement §2–3 of the [README](../README.md). Equation numbers
refer to Tiblias et al. (2025), [arXiv:2504.10542](https://arxiv.org/abs/2504.10542).

## 1. Quadratic Pauli features

For an input $x \in \mathbb{R}^{2^n}$ and an $n$-qubit Pauli string $P$, the interaction
feature is

$$\phi_P(x) = x^\top P x .$$

Pauli strings with an odd number of $Y$ factors are purely imaginary, so $\phi_P \equiv 0$
for real $x$ ([`is_dead_string`](../src/generators/spectral_pauli_generator.py)).
Strings are written most-significant qubit first, so `'XZ'` is $X \otimes Z$; the same
convention is used by [`get_pauli_tensor`](../src/utils/pauli_utils.py), the PennyLane
observables and the FWHT decomposition.

## 2. Basis selection

| Method | Score | Module |
| :--- | :--- | :--- |
| Spectral (default) | $\lvert c_P \rvert$, $c_P = 2^{-n}\,\mathrm{Tr}(\Delta P)$ | [`spectral_pauli_generator.py`](../src/generators/spectral_pauli_generator.py) |
| Spectral via FWHT | same coefficients in $O(4^n n)$ | [`fwht_sim_classifier.py`](../src/models/fwht_sim_classifier.py), [`pauli_decompose.py`](../src/utils/pauli_decompose.py) |
| Cauchy–Schwarz QMI | $\log(V_J V_M / V_C^2)$ between $\phi_P(x)$ and $y$ | [`qmi_pauli_generator.py`](../src/generators/qmi_pauli_generator.py) |
| Random | uniform over the full or live basis | inlined in the experiments |

$\Delta$ is the difference of class-conditional covariances (`moment='covariance'`,
default) or of second moments (`moment='second_moment'`). The two differ whenever the
class means are non-zero; see the [CHANGELOG](../CHANGELOG.md) for the measured effect.

`get_adaptive_spectral_paulis(eta=...)` keeps the smallest prefix whose cumulative
$\lvert c_P \rvert$ reaches a share $\eta$ of the total. The FWHT path accumulates
$\lvert c_P \rvert^2$ instead, so the same $\eta$ keeps a different number of strings.

Selection must only ever see the training split. The self-selecting models
(`SpectralSIMClassifier`, `SpectralExactSIMClassifier`, `FWHTSIMClassifier`,
`FWHTExactSIMClassifier`) enforce this by selecting inside `fit()`.

## 3. Models

| Model | Decision function | Parameters | Module |
| :--- | :--- | :--- | :--- |
| `SIMClassifier` | logistic regression on standardised $\phi_{P_j}(x)$ | $p$ | [`sim_classifier.py`](../src/models/sim_classifier.py) |
| `ExactSIMClassifier` | Eq. 9: $\sigma\big(\sum_j \phi_{P_j}(\tilde x)\, w_j \langle P_j\rangle_\theta\big)$ | $p + 2^n + 3Ln$ | [`exact_sim_classifier.py`](../src/models/exact_sim_classifier.py) |
| `NISQSIMClassifier` | Eq. 9 with gate, $T_1/T_2$ and readout noise | as above | [`nisq_sim_classifier.py`](../src/models/nisq_sim_classifier.py) |
| `HAMClassifier` | Eq. 2–3: $\sigma(\psi_\theta^\dagger (H^0 + \tilde x\tilde x^\top) \psi_\theta)$ | $4^n + 3Ln$ | [`hamiltonian_variants.py`](../src/models/hamiltonian_variants.py) |
| `PEFFClassifier` | §3.2: $\sigma(a\,\lvert\langle\psi_\theta \vert \tilde x\rangle\rvert^2 + b)$ | $2^n + 2 + 3Ln$ | [`hamiltonian_variants.py`](../src/models/hamiltonian_variants.py) |

Here $\tilde x = \tfrac{1}{s}\sum_i x_i + b_\phi$ (Eq. 5), $L$ is the number of
`StronglyEntanglingLayers`, and $\langle P_j\rangle_\theta$ is evaluated once per batch
because it does not depend on the input. PEFF's Hamiltonian is positive semi-definite as
written, so a learnable scale $a$ and offset $b$ are added to make it trainable.

`ExactSIMClassifier` is trained jointly over the circuit parameters $\theta$, the
re-weighting vector $w$ and the input bias $b_\phi$ with Adam and binary cross-entropy.
Because $w_j \langle P_j\rangle_\theta$ collapses to one free coefficient per string, the
variational state adds no expressivity (README §5.5).

## 4. Noise model

`NISQSIMClassifier` reproduces `StronglyEntanglingLayers` gate for gate (including its
cycling CNOT ranges) on `default.mixed`, inserting

- depolarizing channels after single-qubit (`p_gate_1q`) and two-qubit (`p_gate_2q`) gates;
- amplitude damping with $\gamma = 1 - e^{-t/T_1}$ and pure dephasing with
  $1/T_\varphi = 1/T_2 - 1/(2T_1)$, so the realised coherence time is exactly $T_2$;
- readout misassignment as an attenuation $(1-2p_{\text{ro}})^{\mathrm{wt}(P)}$, which is
  basis-independent.

With every rate set to zero it reproduces `ExactSIMClassifier` to machine precision
(checked in `tests/test_models.py`). Per-gate depolarizing for the self-selecting models
is applied through `src/utils/noise_utils.py`.

## 5. Evaluation protocol

- Splits are 70/30 with a fixed `random_state`; seeds are averaged where stated.
- E. Coli is 13.7% positive: report F1 and balanced accuracy alongside accuracy, and
  compare accuracy with the majority-class baseline (0.8709).
- Use the leak-free loaders (`load_ecoli_split`, `select_topk_chi2`,
  `load_20newsgroups_split`) for any reported number.
- Call `set_seed()` once at the top of a script's `__main__` block; library modules never
  reseed on import.
