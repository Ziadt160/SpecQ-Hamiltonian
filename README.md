# Spectral Pauli Pruning for Flipped Quantum Hamiltonian Classifiers

*Ziad Tarek Mohammed — Evoth Labs*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![builds on arXiv:2504.10542](https://img.shields.io/badge/builds%20on-arXiv%3A2504.10542-B31B1B.svg)](https://arxiv.org/abs/2504.10542)

---

## Abstract

Flipped quantum models encode classical data into the measured *observable* rather than
the quantum state, achieving $O(\log d)$ qubit scaling and avoiding the data-loading
bottleneck of variational quantum classifiers. The cost is a measurement basis that grows
as $4^n$. The originating work (Tiblias et al., 2025) selects the measured subset **at
random**. We ask whether that subset can be chosen deliberately, and find that it can: on
a 2,080-string basis, **24 observables retain 95% of full-basis accuracy — an 86-fold
reduction in measurement cost**. We further show that for real-valued inputs
**approximately half of any Pauli basis is identically zero** and therefore unmeasurable,
that spectral ranking accounts for 52% of its advantage over random selection
independently of that effect, and that the generalisation gap opens at roughly 2% of the
training-set size. Finally, we report a negative result: the variational quantum state in
this architecture contributes **no expressivity**, which we establish empirically across
four experiments and then prove algebraically. We conclude that the value of flipped
Hamiltonian classifiers lies in observable selection, not in the variational circuit.

---

## 1. Introduction

Variational quantum classifiers encode an input $x \in \mathbb{R}^d$ into a state
$|\psi(x)\rangle$, which requires either deep circuits or $O(d)$ qubits. *Flipped models*
(Jerbi et al., 2024) invert this: the data enters the observable, and the quantum device
measures a fixed variational state against it. Tiblias et al. (2025) apply this to
classification with three variants — HAM, PEFF and SIM — reaching $O(\log d)$ qubits.

SIM, the most hardware-frugal variant, truncates the Hamiltonian to $p$ Pauli strings.
Its decision function is

$$f_{\theta,\phi}(x) = \sigma\!\left(\frac{1}{2^n}\sum_{j=1}^{p}
(\tilde x^\top P_j \tilde x)\, w_j \,\langle\psi_\theta|P_j|\psi_\theta\rangle\right)$$

with $\tilde x = \frac{1}{s}\sum_i x_i + b_\phi$. The choice of $\{P_j\}$ is left open;
the paper states they "can be chosen at random."

This work asks three questions:

1. **How few observables suffice?** (§5.1)
2. **Does a principled ranking beat random selection, and why?** (§5.2–5.3)
3. **What does the variational state contribute?** (§5.5)

The third question produced the result we consider most consequential, and it is
negative.

---

## 2. Background

An $n$-qubit Hamiltonian decomposes in the Pauli basis
$\mathcal{P}_n = \{I,X,Y,Z\}^{\otimes n}$ as $H = \sum_P \alpha_P P$, with $4^n$ terms.
SIM constructs $H(\tilde x) = \tilde x \tilde x^\top$, extracts coefficients
$\alpha_j = \frac{1}{2^n}\mathrm{Tr}(P_j H) = \frac{1}{2^n}\tilde x^\top P_j \tilde x$,
and measures only $p \ll 4^n$ of them. Since $\langle P_j \rangle$ does not depend on the
input, the circuit is evaluated **once per batch** rather than once per sample — the
architecture's principal efficiency claim.

---

## 3. Method

### 3.1 Spectral moment selection

Rank each Pauli string by its projection onto the class-conditional difference matrix:

$$\Delta = \mathrm{Cov}(x \mid y{=}1) - \mathrm{Cov}(x \mid y{=}0),
\qquad c_P = \tfrac{1}{2^n}\,\mathrm{Tr}(\Delta P)$$

Strings are ordered by $|c_P|$, and an energy cutoff $\eta$ selects the smallest set
reaching that share of the total spectral mass. Selection is fit on the training split
only. Implemented in
[`spectral_pauli_generator.py`](src/generators/spectral_pauli_generator.py).

**Complexity.** The projection over all $4^n$ strings is a single vectorised contraction
at $O(4^n \cdot 2^{2n})$, replacing a loop of $4^n$ matrix products at
$O(4^n \cdot 2^{3n})$ — bit-identical values, 5.8× faster measured at $n{=}4$, widening as
$2^n$. This is not an eigendecomposition, and no $O(2^{3n})$ route is available.

### 3.2 Structurally dead observables

For real-valued $x$ and a Pauli string $P$ with an **odd number of $Y$ factors**, $P$ is
purely imaginary and $x^\top P x \equiv 0$ exactly. Such an observable returns no
information at any shot budget. Their count is

$$N_{\text{dead}}(n) = \frac{4^n - 2^n}{2} \;\longrightarrow\; \tfrac{1}{2}\,4^n$$

### 3.3 Model variants

All three architectures are implemented. HAM (Eq. 2–3) and PEFF (§3.2) evaluate the full
Hamiltonian expectation from the state vector; SIM (Eq. 9) uses the truncated Pauli sum.

---

## 4. Experimental setup

| Dataset | Type | Raw size | Reduction | Qubits |
| :--- | :--- | :--- | :--- | ---: |
| E. Coli (CTZ resistance) | Bio-informatics | 1,935 × 17,198 genes | $\chi^2$ to $2^n$ | 4 – 6 |
| 20 Newsgroups | NLP | ~1,900 docs, TF-IDF 5,000 | PCA to $2^n$ | 4 – 6 |
| Digits (8×8) | Vision | 1,797 × 64 | none | 6 |
| Wine | Tabular | 178 × 13 | zero-pad to 16 | 4 |

All preprocessing is fit on the training split only. E. Coli is **13.7% positive**, so
accuracy is reported against a majority-class baseline of **0.8709**; F1 and balanced
accuracy are the informative metrics. Results are averaged over 3–5 seeds.

---

## 5. Results

### 5.1 The measured basis compresses by one to two orders of magnitude

Smallest basis retaining a share of full-basis test F1
([`pruning_limits.txt`](results/pruning_limits.txt), 5 seeds):

| Dataset | $n$ | Live basis | Full F1 | $k$ @ 95% | Compression |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 20 Newsgroups | 4 | 136 | 0.7893 | 8 | 17.0× |
| 20 Newsgroups | 6 | 2,080 | 0.8252 | 64 | 32.5× |
| E. Coli | 4 | 136 | 0.7011 | 12 | 11.3× |
| E. Coli | 6 | 2,080 | 0.6883 | **24** | **86.7×** |

Redundancy grows with basis size: the larger the Pauli basis, the greater the fraction
that can be discarded.

### 5.2 Half of any random basis is unmeasurable

| $n$ | 4 | 6 | 8 | 10 |
| :--- | ---: | ---: | ---: | ---: |
| Dead fraction | 46.9% | 49.2% | 49.8% | **49.95%** |

Verified numerically: over 200 random real inputs the largest $|x^\top P x|$ across dead
strings is $6\times10^{-15}$, against a live-string minimum of 17.1. At the reference
paper's tuned configuration ($p{=}1000$, $n{=}10$), roughly **500 of 1,000 measured
observables are identically zero**.

Spectral ranking assigns these $|c_P| \approx 10^{-18}$ and places every one below every
live string — the first dead string appears at rank 136/256 ($n{=}4$) and 2080/4096
($n{=}6$) — so any top-$k$ is 100% live at no additional cost.
([`measurement_efficiency.txt`](results/measurement_efficiency.txt))

### 5.3 The ranking is informative beyond that filter

Comparing spectral against draws from the *full* basis conflates two claims. Controlled
against draws from **live strings only**
([`spectral_vs_random_live.txt`](results/spectral_vs_random_live.txt), 4 configurations ×
5 seeds × 5 draws):

| Component | F1 gain | Share |
| :--- | ---: | ---: |
| Total advantage over naive random selection | +0.118 | 100% |
| — dead-string filter | +0.057 | 48% |
| — **spectral ranking** | **+0.062** | **52%** |

The ranking wins **17 of 20** configurations, is worth **+0.30 F1 at $k{=}8$** on E. Coli
$n{=}6$, and washes out as $k$ approaches the full basis, as it must.

**Independent corroboration.** L1-regularised logistic regression, which has no knowledge
of Pauli structure, selects 118 interactions; **88.98% fall within the spectral top-118**
([`lasso_comparison.txt`](results/lasso_comparison.txt)).

### 5.4 Additional observables buy memorisation, not generalisation

E. Coli $n{=}6$, 1,354 training samples. Each retained string is exactly one parameter:

| $k$ | $k/N$ | Train F1 | Test F1 | Gap |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.01 | 0.5489 | 0.5499 | −0.001 |
| **24** | 0.02 | 0.6475 | **0.6553** | −0.008 |
| 64 | 0.05 | 0.7329 | 0.6506 | +0.082 |
| 128 | 0.09 | 0.8398 | 0.6593 | +0.181 |
| 2,080 | 1.54 | 0.9402 | 0.6883 | +0.252 |

The gap opens at $k \approx 16$–$32$, about **2% of the training set** — far earlier than
the conventional $k \approx N$ heuristic. Growing from 24 to 2,080 strings costs 87× the
measurement budget and yields **+0.033 test F1** while train F1 rises 0.65 → 0.94. Below
$k \approx 12$ the gap is negative, indicating genuine underfitting: there is a floor.

### 5.5 Negative result: the variational state contributes no expressivity

Because $\langle P_j \rangle$ is input-independent and $w_j$ is unconstrained, the product
$w_j\langle P_j\rangle$ is a single free vector. The hypothesis class is therefore
identical to logistic regression on the quadratic features $x^\top P_j x$.

| Evidence | Result |
| :--- | :--- |
| Circuit trained / frozen at random init / removed (5 seeds) | 0.9363 / **0.9373** / 0.9346 — frozen is nominally best |
| HAM vs PEFF vs SIM vs circuit-free SIM | circuit-free matches or wins ([`variant_comparison.txt`](results/variant_comparison.txt)) |
| Shot budget 1k → analytic | all arms saturate at the smallest budget; 20% error per $\langle P\rangle$ costs nothing |
| **Algebraic** | with standardised features, damping $\langle P\rangle$ by $10^{-6}$ or inverting every sign leaves the fitted model **identical to $10^{-12}$** |

The final row is a proof rather than an observation: $e_j$ is constant across samples, so
standardisation divides it out exactly, and a free $w_j$ absorbs the residual sign.

This is consistent with the parent paper's own observations — that removing $w_j$
degrades performance (§3.3), and that non-entangling circuits often perform best (§4.4).
Neither work tested removing $\langle P \rangle$; that is the direction reported here.

### 5.6 Model variants

Head-to-head, 3 seeds, spectral top-32 basis for SIM:

| Variant | E. Coli acc / F1 | 20 News acc / F1 | Params ($n{=}4$) |
| :--- | :--- | :--- | ---: |
| HAM | 0.8663 / 0.000 | 0.5671 / 0.724 | 292 — $O(4^n)$ |
| PEFF | 0.9312 / 0.687 | 0.8089 / 0.835 | **54** — $O(d)$ |
| SIM | **0.9306 / 0.720** | 0.8188 / 0.840 | 84 — $O(p)$ |
| SIM, circuit removed | 0.9283 / 0.706 | **0.8231 / 0.845** | 84 |

PEFF meets its design goal: within one point of SIM at a third of the parameters. HAM
collapsed to a single class (balanced accuracy exactly 0.5000) under shared
hyperparameters; the reference work tunes each variant separately with 8–32 ansatz layers
against the 3 used here, so this is a fair-comparison artifact rather than a verdict on
HAM.

---

## 6. Discussion

**Where the value lies.** The compression and dead-observable results are properties of
*observable selection* and hold independently of any quantum advantage claim. They reduce
measurement cost — the resource that dominates on hardware — by one to two orders of
magnitude. The variational circuit, by contrast, is shown to be inert.

**On noise.** Depolarizing noise damps $e_j \to \lambda_j e_j$ with $\lambda_j > 0$;
since $w_j$ is free, $w_j' = w_j/\lambda_j$ recovers the identical function. Under global
depolarizing, $\mathrm{sign}(\text{logit})$ — and hence accuracy — is **exactly**
invariant. Three experiments (shot budget, pruning under noise, penalised pruning) found
no measurable effect at rates up to $p{=}0.30$. We therefore make **no noise-robustness
claim**: an apparently flat degradation curve here reflects the parametrisation, not
hardware resilience. Pruning reduces measurement cost linearly; it does not make the
circuit quieter.

**On the E. Coli benchmark.** Reducing to $2^4{=}16$ genes leaves 61% of samples as
all-zero vectors, which any model of the form $f(x^\top P x)$ must map to one constant
prediction; the effect disappears by 64 genes. Plain logistic regression on 256 genes
reaches F1 0.812, above every Pauli configuration tested. On this dataset the quadratic
map does not add discriminative power
([`ecoli_feature_count.txt`](results/ecoli_feature_count.txt)).

---

## 7. Limitations

- Experiments run at $n \in \{4,6\}$. The reference work operates at $n \in \{9,10\}$,
  where $p{=}1000$ is under 0.4% of the live basis — a regime our sweeps do not reach.
- SST2 and full MNIST are not implemented: both require sequence inputs ($s>1$) and a
  basis beyond what a dense Pauli tensor holds.
- The covariance and second-moment definitions of $\Delta$ select substantially different
  bases (21–25% overlap at $k{=}16$) but are **statistically indistinguishable on
  accuracy**; the default follows the paper on principle, not on measured performance.
- `experiment_nisq_sweep.py` emits only a figure, with no machine-readable output.
- The overfitting comparison between spectral and random orderings is evaluated at the
  full basis, where both contain the same strings; that comparison is degenerate.

Further known issues are recorded in [CHANGELOG.md](CHANGELOG.md).

---

## 8. Reproducibility

```bash
git clone https://github.com/Ziadt160/SpecQ-Hamiltonian.git
```

Install the package, not merely its requirements — experiments import `src.*`:

```bash
pip install -e .
```

Run from the repository root; all output is written to `results/`.

```bash
python experiments/experiment_pruning_limits.py
```

| Experiment | Question | Section |
| :--- | :--- | :--- |
| [`pruning_limits`](experiments/experiment_pruning_limits.py) | How far can the basis be cut? Where does overfitting begin? | 5.1, 5.4 |
| [`measurement_efficiency`](experiments/experiment_measurement_efficiency.py) | How much of a random basis is structurally unmeasurable? | 5.2 |
| [`spectral_vs_random_live`](experiments/experiment_spectral_vs_random_live.py) | Is the ranking informative, or only a dead-string filter? | 5.3 |
| [`circuit_ablation`](experiments/experiment_circuit_ablation.py) | Does the variational state add expressivity? | 5.5 |
| [`variant_comparison`](experiments/experiment_variant_comparison.py) | HAM vs PEFF vs SIM under identical conditions | 5.6 |
| [`shot_budget`](experiments/experiment_shot_budget.py) | Does finite sampling favour a smaller basis? | 6 |
| [`delta_definition`](experiments/experiment_delta_definition.py) | Covariance vs second moment | 7 |
| [`ecoli_feature_count`](experiments/experiment_ecoli_feature_count.py) | Is the $2^n$ gene budget an artifact? | 6 |

### Repository layout

| Path | Contents |
| :--- | :--- |
| [`src/models/`](src/models) | SIM (Eq. 9), HAM and PEFF, NISQ variant, classical baseline |
| [`src/generators/`](src/generators) | Spectral moment selection, Cauchy-Schwarz QMI |
| [`src/utils/`](src/utils) | Pauli algebra, leak-free data loaders |
| [`src/analysis/`](src/analysis) | Interaction geometry, canonical patterns, stress tests |
| [`experiments/`](experiments) | Benchmark and ablation suites |
| [`results/`](results) | Numerical output and figures |

> [!IMPORTANT]
> `load_ecoli_split` and `load_20newsgroups_split` fit all preprocessing on the training
> split and should be used for any reported metric. `load_ecoli_reduced` fits $\chi^2$
> selection on the full dataset including test labels; it is retained only for
> non-evaluative geometry analysis and warns at call time.

---

## References

1. F. Tiblias, A. Schroeder, Y. Zhang, M. Gachechiladze, I. Gurevych. *An Efficient
   Quantum Classifier Based on Hamiltonian Representations.*
   [arXiv:2504.10542](https://arxiv.org/abs/2504.10542), 2025.
2. S. Jerbi, C. Gyurik, S. C. Marshall, R. Molteni, V. Dunjko. *Shadows of quantum machine
   learning.* Nature Communications 15:5676, 2024.
3. M. Cerezo et al. *Variational Quantum Algorithms.* Nature Reviews Physics 3, 2021.
4. Z. T. Mohammed. *Spectral Pauli Pruning: Efficient Feature Selection for Flipped
   Quantum Hamiltonian Classifiers.* [Manuscript](pdf/Spectral_Pauli_Generator.pdf).

---

## License

MIT.
