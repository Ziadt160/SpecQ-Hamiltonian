# Spectral Pauli Pruning for Flipped Quantum Hamiltonian Classifiers

*Ziad Tarek Mohammed — Evoth Labs*

[![tests](https://github.com/Ziadt160/SpecQ-Hamiltonian/actions/workflows/tests.yml/badge.svg)](https://github.com/Ziadt160/SpecQ-Hamiltonian/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Abstract

Flipped quantum models encode classical data into the measured *observable* rather than
the quantum state, achieving $O(\log d)$ qubit scaling and avoiding the data-loading
bottleneck of variational quantum classifiers. The cost is a measurement basis that grows
as $4^n$. The originating work (Tiblias et al., 2025) selects the measured subset **at
random**. We ask whether that subset can be chosen deliberately, and find that it can: on
a 2,080-string basis, **24 observables retain 95% of full-basis F1 — an 86-fold
reduction in measurement cost**. We further show that for real-valued inputs
**approximately half of any Pauli basis is identically zero** and therefore unmeasurable,
that spectral ranking accounts for 52% of its advantage over random selection
independently of that effect, and that the generalisation gap opens at roughly 2% of the
training-set size. Finally, we report a negative result: the variational quantum state in
this architecture contributes **no expressivity**, which we establish empirically through
circuit ablations and then prove algebraically. We conclude that the value of flipped
Hamiltonian classifiers lies in observable selection, not in the variational circuit.

### Highlights

| Finding | Evidence |
| :--- | :--- |
| 24 of 2,080 observables keep 95% of full-basis F1 (86.7× fewer measurements) | [§5.1](#51-the-measured-basis-compresses-by-one-to-two-orders-of-magnitude) |
| ~50% of any Pauli basis is identically zero on real inputs | [§5.2](#52-half-of-any-random-basis-is-unmeasurable) |
| Spectral ranking beats random-among-live strings in 17/20 configurations | [§5.3](#53-the-ranking-is-informative-beyond-that-filter) |
| The variational state is a redundant reparametrisation | [§5.5](#55-negative-result-the-variational-state-contributes-no-expressivity) |

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
the parent paper selects them at random.

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

**Complexity.** The dense projection over all $4^n$ strings is a single vectorised
contraction at $O(4^n \cdot 2^{2n})$, replacing a loop of $4^n$ matrix products at
$O(4^n \cdot 2^{3n})$ — bit-identical values, 5.8× faster measured at $n{=}4$. For larger
$n$, [`pauli_decompose.py`](src/utils/pauli_decompose.py) computes the same coefficients
with a fast Walsh–Hadamard transform in $O(4^n \cdot n)$ (Georges et al., 2025); the test
suite checks that the two back-ends agree string by string.

### 3.2 Structurally dead observables

For real-valued $x$ and a Pauli string $P$ with an **odd number of $Y$ factors**, $P$ is
purely imaginary and $x^\top P x \equiv 0$ exactly. Such an observable returns no
information at any shot budget. Their count is

$$N_{\text{dead}}(n) = \frac{4^n - 2^n}{2} \;\longrightarrow\; \tfrac{1}{2}\,4^n$$

### 3.3 Model variants

All three architectures are implemented. HAM (Eq. 2–3) and PEFF (§3.2) evaluate the full
Hamiltonian expectation from the state vector; SIM (Eq. 9) uses the truncated Pauli sum.
A noisy SIM with gate depolarisation, $T_1/T_2$ relaxation and readout error is provided
for hardware-realistic studies; with all noise switched off it reproduces the noiseless
model exactly.

---

## 4. Experimental setup

| Dataset | Type | Raw size | Reduction | Qubits |
| :--- | :--- | :--- | :--- | ---: |
| E. Coli (CTZ resistance) | Bio-informatics | 1,935 × 17,198 genes | $\chi^2$ to $2^n$ | 4 – 6 |
| 20 Newsgroups | NLP | ~1,800 docs, TF-IDF 5,000 | PCA to $2^n$ | 4 – 6 |
| Digits (8×8) | Vision | 1,797 × 64 | none | 6 |
| Wine | Tabular | 178 × 13 | zero-pad to 16 | 4 |

Everything that sees labels — $\chi^2$ gene selection, spectral ranking, model fitting —
is fit on the training split only. The 20 Newsgroups TF-IDF/PCA projection used by the
main experiments is fit on the whole corpus without labels (transductive);
`load_20newsgroups_split` is the strictly inductive alternative. E. Coli is **13.7%
positive**, so accuracy is reported against a majority-class baseline of **0.8709**; F1
and balanced accuracy are the informative metrics. Results are averaged over 3–5 seeds.

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
| E. Coli | 6 | 2,080 | 0.6897 | **24** | **86.7×** |

Redundancy grows with basis size: the larger the Pauli basis, the greater the fraction
that can be discarded.

### 5.2 Half of any random basis is unmeasurable

| $n$ | 4 | 6 | 8 | 10 |
| :--- | ---: | ---: | ---: | ---: |
| Dead fraction | 46.9% | 49.2% | 49.8% | **49.95%** |

Verified numerically: over 200 random real inputs at $n{=}5$ the largest $|x^\top P x|$
across dead strings is $6\times10^{-15}$, against a live-string minimum of 17.1. At the
reference paper's tuned configuration ($p{=}1000$, $n{=}10$), roughly **500 of 1,000
measured observables are identically zero**.

Spectral ranking assigns these $|c_P| \approx 10^{-18}$ and places every one below every
live string — on both datasets all 136 ($n{=}4$) and all 2,080 ($n{=}6$) live strings
rank ahead of the first dead one — so any top-$k$ is 100% live at no additional cost
([`measurement_efficiency.txt`](results/measurement_efficiency.txt)).

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

**A comparison that does not discriminate.** L1-regularised logistic regression, which
has no knowledge of Pauli structure, selects 118 interactions at $n{=}4$, and 86.4% of
them fall within the spectral top-118. A random 118 of the 135 non-constant strings
overlaps by 86.7%, so at this basis size the agreement is at chance level and we do not
count it as corroboration
([`lasso_comparison.txt`](results/lasso_comparison.txt)). A meaningful version needs a
basis large enough that L1 selects a small fraction of it.

### 5.4 Additional observables buy memorisation, not generalisation

E. Coli $n{=}6$, 1,354 training samples. Each retained string is exactly one parameter:

| $k$ | $k/N$ | Train F1 | Test F1 | Gap |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.01 | 0.5489 | 0.5499 | −0.001 |
| **24** | 0.02 | 0.6475 | **0.6553** | −0.008 |
| 64 | 0.05 | 0.7334 | 0.6506 | +0.083 |
| 128 | 0.09 | 0.8403 | 0.6593 | +0.181 |
| 2,080 | 1.54 | 0.9402 | 0.6897 | +0.251 |

The gap opens at $k \approx 16$–$32$, about **2% of the training set** — far earlier than
the conventional $k \approx N$ heuristic. Growing from 24 to 2,080 strings costs 87× the
measurement budget and yields **+0.034 test F1** while train F1 rises 0.65 → 0.94. Below
$k \approx 12$ the gap is negative, indicating genuine underfitting: there is a floor.

### 5.5 Negative result: the variational state contributes no expressivity

Because $\langle P_j \rangle$ is input-independent and $w_j$ is unconstrained, the product
$w_j\langle P_j\rangle$ is a single free vector. The hypothesis class is therefore
identical to logistic regression on the quadratic features $x^\top P_j x$.

| Evidence | Result |
| :--- | :--- |
| Circuit trained / frozen at random init / removed (5 seeds) | 0.9394 / 0.9377 / 0.9387 accuracy — all three within one standard deviation (±0.003–0.012) ([`circuit_ablation.txt`](results/circuit_ablation.txt)) |
| HAM vs PEFF vs SIM vs circuit-free SIM | circuit-free matches or wins ([`variant_comparison.txt`](results/variant_comparison.txt)) |
| Shot budget 1k → analytic | all arms saturate at the smallest budget — as the next row predicts, since an estimation error in $\langle P_j\rangle$ is one constant per feature ([`shot_budget.txt`](results/shot_budget.txt)) |
| **Algebraic** | with standardised features, damping $\langle P\rangle$ by $10^{-6}$ or inverting every sign leaves the fitted model **identical to $10^{-12}$** |

The final row is a proof rather than an observation: $e_j$ is constant across samples, so
standardisation divides it out exactly, and a free $w_j$ absorbs the residual sign.

This is consistent with the parent paper's own observations — that removing $w_j$
degrades performance (§3.3), and that non-entangling circuits often perform best (§4.4).
Neither work tested removing $\langle P \rangle$; that is the direction reported here.

### 5.6 Model variants

Head-to-head, 3 seeds, spectral top-32 basis for SIM
([`variant_comparison.txt`](results/variant_comparison.txt)):

| Variant | E. Coli acc / F1 | 20 News acc / F1 | Params ($n{=}4$) |
| :--- | :--- | :--- | ---: |
| HAM | 0.8663 / 0.000 | 0.5671 / 0.724 | 292 — $O(4^n)$ |
| PEFF | 0.9312 / 0.687 | 0.8089 / 0.835 | 54 — $O(d)$ |
| SIM | 0.9317 / 0.727 | 0.8207 / 0.840 | 84 — $O(p)$ |
| SIM, circuit removed | **0.9329 / 0.734** | **0.8231 / 0.844** | **48** |

Removing the circuit leaves the best model on both datasets with the fewest parameters,
as §5.5 predicts. PEFF comes within about one accuracy point of SIM with a third fewer
parameters, at some cost in F1 on E. Coli. HAM collapsed to a single class (balanced
accuracy exactly 0.5000) under shared hyperparameters; the reference work tunes each
variant separately with 8–32 ansatz layers against the 3 used here, so this is a
fair-comparison artifact rather than a verdict on HAM.

---

## 6. Discussion

**Where the value lies.** The compression and dead-observable results are properties of
*observable selection* and hold independently of any quantum advantage claim. They reduce
measurement cost — the resource that dominates on hardware — by one to two orders of
magnitude. The variational circuit, by contrast, is shown to be inert.

**On noise.** Depolarizing noise damps $e_j \to \lambda_j e_j$ with $\lambda_j > 0$;
since $w_j$ is free, $w_j' = w_j/\lambda_j$ recovers the identical function. Under global
depolarizing, $\mathrm{sign}(\text{logit})$ — and hence accuracy — is **exactly**
invariant. Two of our noise experiments (shot budget, pruning under noise) standardise
the features *after* multiplying by the estimated $\langle P_j\rangle$, which divides any
per-string damping or shot error out exactly; their flat curves restate this algebra and
are not independent evidence. The third
([`penalised_pruning.txt`](results/penalised_pruning.txt)) fixes the feature scaling and
penalises $w$, so compensating for noise has a cost. There, noise does bite: with a
strong penalty ($C{=}0.01$) E. Coli F1 falls from 0.625 at $p{=}0$ to 0 at $p \ge 0.2$
and 20 Newsgroups from 0.793 to 0.724 at $p{=}0.3$, while at $C \ge 0.1$ the loss is at
most 0.06. Noise costs accuracy exactly to the extent that the weights are prevented from
rescaling. We therefore make **no noise-robustness claim**: a flat degradation curve in
this architecture reflects the parametrisation, not hardware resilience. Pruning reduces
measurement cost linearly; it does not make the circuit quieter.

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
- The overfitting comparison between spectral and random orderings is evaluated at the
  full basis, where both contain the same strings; that comparison is degenerate.
- Every result cited here was regenerated with the current code (v0.2.0). A few
  long-running auxiliary experiments were not; they, and known issues in individual
  scripts, are listed in [CHANGELOG.md](CHANGELOG.md#known-limitations).

---

## 8. Reproducibility

### Installation

```bash
git clone https://github.com/Ziadt160/SpecQ-Hamiltonian.git
cd SpecQ-Hamiltonian
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Install the package, not merely its requirements — experiments import `src.*`. Python
3.10+ is required; the code is tested with PyTorch 2.9 and PennyLane 0.42. A GPU is not
needed: every experiment runs on CPU.

### Data

| Dataset | How to obtain |
| :--- | :--- |
| E. Coli | Included: [`data/EColi_Merged_df.csv`](data/EColi_Merged_df.csv), gene presence/absence and resistance phenotypes for 1,936 isolates from Moradigaravand et al. (2018). |
| 20 Newsgroups | Downloaded automatically by scikit-learn on first use (~14 MB, cached). |
| Digits, Wine | Bundled with scikit-learn. |
| Credit-card fraud (case study only) | Download `creditcard_2023.csv` from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) into `data/`. It is not committed (325 MB). |

### Running experiments

Run from the repository root; all output is written to `results/`, whatever the working
directory.

```bash
python -m experiments.experiment_pruning_limits
```

| Experiment | Question | Section |
| :--- | :--- | :--- |
| [`pruning_limits`](experiments/experiment_pruning_limits.py) | How far can the basis be cut? Where does overfitting begin? | 5.1, 5.4 |
| [`measurement_efficiency`](experiments/experiment_measurement_efficiency.py) | How much of a random basis is structurally unmeasurable? | 5.2 |
| [`spectral_vs_random_live`](experiments/experiment_spectral_vs_random_live.py) | Is the ranking informative, or only a dead-string filter? | 5.3 |
| [`lasso`](experiments/experiment_lasso.py) | Does an L1 model with no Pauli knowledge pick the same strings? | 5.3 |
| [`circuit_ablation`](experiments/experiment_circuit_ablation.py) | Does the variational state add expressivity? | 5.5 |
| [`variant_comparison`](experiments/experiment_variant_comparison.py) | HAM vs PEFF vs SIM under identical conditions | 5.6 |
| [`shot_budget`](experiments/experiment_shot_budget.py) | Does finite sampling favour a smaller basis? | 6 |
| [`ecoli_feature_count`](experiments/experiment_ecoli_feature_count.py) | Is the $2^n$ gene budget an artifact? | 6 |
| [`delta_definition`](experiments/experiment_delta_definition.py) | Covariance vs second moment | 7 |

Every other script — dataset benchmarks, noise studies, baselines and the credit-card
fraud case study — is catalogued in [docs/experiments.md](docs/experiments.md).

### Tests

```bash
pytest
```

The suite (~30 s on CPU) checks the claims the results rest on rather than only the
plumbing: dead strings vanish exactly, spectral and FWHT selection agree, the noisy model
reduces to the exact one with noise off, HAM/PEFF parameter counts, that each synthetic
regime is solved by its own Pauli class, and that selection never sees test labels.

### Repository layout

| Path | Contents |
| :--- | :--- |
| [`src/models/`](src/models) | SIM (Eq. 9), HAM and PEFF, noisy SIM, self-selecting SIM, VQC / QSVM baselines |
| [`src/generators/`](src/generators) | Spectral moment selection, Cauchy–Schwarz QMI selection |
| [`src/utils/`](src/utils) | Pauli algebra, FWHT decomposition, leak-free data loaders, seeding, paths |
| [`src/analysis/`](src/analysis) | Interaction geometry, canonical patterns, stress tests |
| [`experiments/`](experiments) | Benchmark and ablation scripts |
| [`results/`](results) | Numerical output and figures |
| [`tests/`](tests) | Unit and property tests |
| [`docs/`](docs) | Methodology notes and the experiment catalogue |

> [!IMPORTANT]
> `load_ecoli_split` and `load_20newsgroups_split` fit all preprocessing on the training
> split and should be used for any reported metric. `load_ecoli_reduced` fits $\chi^2$
> selection on the full dataset including test labels; it is retained only for
> non-evaluative geometry analysis and warns at call time.

---

## Citation

If you use this code, please cite it together with the work it builds on
([CITATION.cff](CITATION.cff) has the machine-readable form):

```bibtex
@software{mohammed_spectral_pauli_pruning,
  author = {Mohammed, Ziad Tarek},
  title  = {Spectral Pauli Pruning for Flipped Quantum Hamiltonian Classifiers},
  url    = {https://github.com/Ziadt160/SpecQ-Hamiltonian},
  year   = {2026}
}

@article{tiblias2025hamiltonian,
  author  = {Tiblias, Federico and Schroeder, Anna and Zhang, Yue and
             Gachechiladze, Mariami and Gurevych, Iryna},
  title   = {An Efficient Quantum Classifier Based on Hamiltonian Representations},
  journal = {arXiv preprint arXiv:2504.10542},
  year    = {2025}
}
```

---

## References

1. F. Tiblias, A. Schroeder, Y. Zhang, M. Gachechiladze, I. Gurevych. *An Efficient
   Quantum Classifier Based on Hamiltonian Representations.*
   [arXiv:2504.10542](https://arxiv.org/abs/2504.10542), 2025.
2. S. Jerbi, C. Gyurik, S. C. Marshall, R. Molteni, V. Dunjko. *Shadows of quantum machine
   learning.* Nature Communications 15:5676, 2024.
3. M. Cerezo et al. *Variational Quantum Algorithms.* Nature Reviews Physics 3, 2021.
4. D. Moradigaravand, M. Palm, A. Farewell, V. Mustonen, J. Warringer, L. Parts.
   *Prediction of antibiotic resistance in Escherichia coli from large-scale pan-genome
   data.* PLoS Computational Biology 14(12):e1006258, 2018.
5. T. N. Georges, B. K. Berntson, C. Sünderhauf, A. V. Ivanov. *Pauli Decomposition via
   the Fast Walsh-Hadamard Transform.* New Journal of Physics, 2025.
   [arXiv:2408.06206](https://arxiv.org/abs/2408.06206).

---

## License

[MIT](LICENSE).
