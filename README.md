# SpecQ-Hamiltonian — Spectral Pauli Pruning for Flipped Quantum Classifiers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![arXiv:2504.10542](https://img.shields.io/badge/builds%20on-arXiv%3A2504.10542-B31B1B.svg)](https://arxiv.org/abs/2504.10542)

**How few quantum observables does a Hamiltonian classifier actually need?**

Flipped quantum models encode data into the *observable* rather than the state, reaching
$O(\log d)$ qubits. But the Pauli basis grows as $4^n$, and the parent work
([Tiblias et al., 2025](https://arxiv.org/abs/2504.10542)) selects the measured subset
**at random**. This project asks whether that subset can be chosen well — and finds that
it can, by a very large margin.

**Headline: 24 of 2,080 observables retain 95% of full-basis accuracy — an 86× reduction
in measurement cost.**

---

## Results

All figures are re-measured from this codebase with leak-free train/test splits and
multiple seeds. E. Coli is 13.7% positive, so accuracy is reported against a
**majority-class baseline of 0.8709**, with F1 and balanced accuracy as the informative
metrics.

### 1. The measured basis can be cut by 1–2 orders of magnitude

Smallest basis retaining a share of full-basis test F1 ([pruning_limits.txt](results/pruning_limits.txt), 5 seeds):

| Dataset | Qubits | Live basis | Full-basis F1 | k for 95% | Compression |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 20 Newsgroups | 4 | 136 | 0.7893 | 8 | **17.0×** |
| 20 Newsgroups | 6 | 2,080 | 0.8252 | 64 | **32.5×** |
| E. Coli | 4 | 136 | 0.7011 | 12 | **11.3×** |
| E. Coli | 6 | 2,080 | 0.6883 | **24** | **86.7×** |

Redundancy grows with basis size — the larger the Pauli basis, the more of it is wasted.

### 2. Half of any random Pauli basis is unmeasurable

For real-valued inputs, a Pauli string with an **odd number of $Y$ factors** has
$x^\top P x \equiv 0$ — exactly zero, not small. Measuring it returns nothing regardless
of shot budget. The count is $(4^n - 2^n)/2$:

| $n$ | 4 | 6 | 8 | 10 |
| :--- | ---: | ---: | ---: | ---: |
| dead fraction | 46.9% | 49.2% | 49.8% | **49.95%** |

At the parent paper's tuned setting ($p{=}1000$ strings, $n{=}10$), roughly **500 of
1,000 measured observables are identically zero**. Spectral ranking assigns them
$|c_P| \approx 10^{-18}$ and places every one *below* every live string — first dead
string at rank 136/256 ($n{=}4$) and 2080/4096 ($n{=}6$) — so any top-$k$ is 100% live at
no cost. ([measurement_efficiency.txt](results/measurement_efficiency.txt))

### 3. The ranking carries information beyond that filter

Spectral is usually compared against strings drawn from the *whole* basis, which
conflates "don't measure zeros" with "rank the rest well". Controlled against draws from
**live strings only** ([spectral_vs_random_live.txt](results/spectral_vs_random_live.txt)):

| Component | F1 gain | Share |
| :--- | ---: | ---: |
| Total advantage over a naive random draw | +0.118 | 100% |
| — dead-string filter | +0.057 | 48% |
| — **the ranking itself** | **+0.062** | **52%** |

The ranking wins **17 of 20** configurations and is worth **+0.30 F1 at $k{=}8$** on
E. Coli $n{=}6$. It matters most under aggressive pruning and washes out as $k$
approaches the full basis, as it must.

### 4. Beyond ~24 observables, extra strings buy memorisation

E. Coli $n{=}6$, 1,354 training samples. Each retained string is exactly one parameter:

| k | k/N | train F1 | test F1 | gap |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.01 | 0.5489 | 0.5499 | −0.001 |
| **24** | 0.02 | 0.6475 | **0.6553** | −0.008 |
| 64 | 0.05 | 0.7329 | 0.6506 | +0.082 |
| 128 | 0.09 | 0.8398 | 0.6593 | +0.181 |
| 2,080 | 1.54 | 0.9402 | 0.6883 | +0.252 |

The generalisation gap opens at $k \approx 16$–$32$ — **2% of the training set**, far
earlier than the usual $k \approx N$ rule of thumb. Going from 24 to 2,080 strings costs
87× the measurement budget and buys **+0.033 test F1** while train F1 climbs 0.65 → 0.94.
Below $k \approx 12$ the gap turns negative: genuine underfitting, so there is a floor.

### 5. Negative result — the variational state contributes nothing

The decision function is
$f(x) = \sigma\big(\sum_j (x^\top P_j x)\, w_j \langle\psi_\theta|P_j|\psi_\theta\rangle\big)$.
Because $\langle P_j \rangle$ does not depend on the input and $w_j$ is free, the product
$w_j \langle P_j\rangle$ is a single unconstrained vector — the quantum branch is a
reparametrisation, not added expressivity.

Confirmed four independent ways:

| Test | Result |
| :--- | :--- |
| Circuit trained / frozen at random init / deleted (5 seeds) | 0.9363 / **0.9373** / 0.9346 — frozen is nominally best |
| HAM vs PEFF vs SIM vs SIM-without-circuit | circuit-free matches or wins ([variant_comparison.txt](results/variant_comparison.txt)) |
| Shot budget 1k → analytic | every arm saturates at the smallest budget; 20% error on every $\langle P\rangle$ costs nothing |
| **Algebraic** | with standardised features, damping $\langle P\rangle$ by $10^{-6}$ or flipping every sign leaves the fitted model **identical to $10^{-12}$** |

The last line is a proof, not an observation: $e_j$ is constant across samples, so
`StandardScaler` divides it out exactly, and a free $w_j$ absorbs the residual sign.

> [!NOTE]
> **On noise.** Depolarizing noise damps $e_j \to \lambda_j e_j$ with $\lambda_j > 0$, and
> $w_j' = w_j/\lambda_j$ recovers the identical function — under *global* depolarizing,
> $\mathrm{sign}(\text{logit})$ and therefore accuracy are **exactly** invariant. Three
> separate experiments (shot budget, pruning-under-noise, penalised pruning) found no
> measurable noise effect. This architecture has no noise-robustness claim to make, and
> the repository states so rather than reporting the flat curve as resilience. Pruning
> reduces **measurement cost** linearly; it does not make the circuit quieter.

---

## Method

Rank every Pauli string by how much it contributes to the class-conditional difference
matrix, then keep the top $k$.

$$\Delta = \mathrm{Cov}(x \mid y{=}1) - \mathrm{Cov}(x \mid y{=}0), \qquad
c_P = \tfrac{1}{2^n}\,\mathrm{Tr}(\Delta P)$$

Strings are ordered by $|c_P|$; an energy cutoff $\eta$ selects the smallest set reaching
that share of the total. Implemented in
[spectral_pauli_generator.py](src/generators/spectral_pauli_generator.py).

**Cost.** The projection onto all $4^n$ strings uses a single vectorised contraction at
$O(4^n \cdot 2^{2n})$, replacing a loop of $4^n$ matrix products at $O(4^n \cdot 2^{3n})$
— bit-identical values, 5.8× faster measured at $n{=}4$ and widening as $2^n$.

**Independent corroboration.** L1-regularised logistic regression, which knows nothing
about Pauli structure, selects 118 interactions — **88.98% of them fall in the spectral
top-118** ([lasso_comparison.txt](results/lasso_comparison.txt)).

> [!IMPORTANT]
> **`moment=` parameter.** `'covariance'` (default) follows the paper; `'second_moment'`
> reproduces the original implementation, which omitted mean subtraction. The two select
> substantially different bases (21–25% overlap at $k{=}16$, converging to 94% by
> $k{=}128$) but are **statistically indistinguishable on accuracy** — a controlled
> 5-seed comparison gives +0.040 at $k{=}8$, +0.074 at $k{=}16$, −0.041 at $k{=}128$,
> against per-seed standard deviations of 0.02–0.08. The default is chosen on principle,
> not on measured performance.

---

## Model variants

All three architectures from the parent paper are implemented, with measured parameter
counts reproducing its stated scaling.

| Variant | Hamiltonian | Parametrisation | Params ($n{=}4$) |
| :--- | :--- | :--- | ---: |
| **HAM** | full, $\psi^\dagger H \psi$ | matrix bias $H^0_\phi$ | 280 — $O(4^n)$ |
| **PEFF** | full, $\psi^\dagger H \psi$ | input bias $b_\phi$ | **42** — $O(d)$ |
| **SIM** | truncated to $p$ strings | Pauli weights $w_j$ | $O(p)$ |

Head-to-head, 3 seeds, spectral top-32 basis for SIM
([variant_comparison.txt](results/variant_comparison.txt)):

| Variant | E. Coli acc / F1 | 20 News acc / F1 | Params |
| :--- | :--- | :--- | ---: |
| HAM | 0.8663 / 0.000 | 0.5671 / 0.724 | 292 |
| PEFF | 0.9312 / 0.687 | 0.8089 / 0.835 | **54** |
| SIM | **0.9306 / 0.720** | 0.8188 / 0.840 | 84 |
| SIM, circuit removed | 0.9283 / 0.706 | **0.8231 / 0.845** | 84 |

PEFF delivers its promise — within a point of SIM at a third of the parameters. HAM
collapsed to a single class (balanced accuracy exactly 0.5000) under shared
hyperparameters; the parent paper tunes each variant separately with 8–32 ansatz layers
against the 3 used here, so this is a fair-comparison artifact rather than a verdict.

Two caveats found during implementation, documented rather than hidden:

- **PEFF's Hamiltonian is positive semi-definite** as written (a sum of rank-1 outer
  products), so $\psi^\dagger H \psi \ge 0$ always and a fixed 0.5 threshold emits one
  class for every input. `PEFFClassifier` adds a single learnable scalar offset.
- **`load_ecoli_reduced` leaks**: it fits chi2 selection on the full dataset including
  test labels. It is retained only for non-evaluative geometry analysis and warns at
  call time. Use `load_ecoli_split` for anything reported.

---

## Getting started

```bash
git clone https://github.com/Ziadt160/SIM-Flipped-Models.git
```

Install the package, not just the requirements — the experiments import `src.*` as a
package:

```bash
pip install -e .
```

Run from the repository root; everything writes to `results/`.

```bash
python experiments/experiment_pruning_limits.py
```

The two experiments most worth reading first:

```bash
python experiments/experiment_measurement_efficiency.py
```
```bash
python experiments/experiment_circuit_ablation.py
```

---

## Repository layout

| Path | Contents |
| :--- | :--- |
| [`src/models/`](src/models) | `exact_sim_classifier.py` (SIM, Eq. 9), `hamiltonian_variants.py` (HAM, PEFF), `nisq_sim_classifier.py`, `sim_classifier.py` |
| [`src/generators/`](src/generators) | Spectral moment selection (Algorithm 3) and Cauchy-Schwarz QMI |
| [`src/utils/`](src/utils) | Pauli algebra and leak-free data loaders |
| [`src/analysis/`](src/analysis) | Interaction geometry, canonical patterns, stress tests |
| [`experiments/`](experiments) | 33 benchmark and ablation suites |
| [`results/`](results) | Numerical output (`.txt`, `.csv`) and figures |

**Key experiments**

| Experiment | Question |
| :--- | :--- |
| [`pruning_limits`](experiments/experiment_pruning_limits.py) | How far can the basis be cut, and where does overfitting start? |
| [`measurement_efficiency`](experiments/experiment_measurement_efficiency.py) | How much of a random basis is structurally unmeasurable? |
| [`spectral_vs_random_live`](experiments/experiment_spectral_vs_random_live.py) | Is the ranking informative, or just a dead-string filter? |
| [`circuit_ablation`](experiments/experiment_circuit_ablation.py) | Does the variational state add expressivity? |
| [`variant_comparison`](experiments/experiment_variant_comparison.py) | HAM vs PEFF vs SIM under identical conditions |
| [`delta_definition`](experiments/experiment_delta_definition.py) | Covariance vs second moment |
| [`shot_budget`](experiments/experiment_shot_budget.py) | Does finite sampling favour a smaller basis? |
| [`ecoli_feature_count`](experiments/experiment_ecoli_feature_count.py) | Is the $2^n$ gene budget an artifact? |

---

## Datasets

| Dataset | Type | Raw size | Reduced to | Qubits |
| :--- | :--- | :--- | :--- | ---: |
| E. Coli (CTZ resistance) | Bio-informatics | 1,935 × 17,198 genes | $2^n$ via chi2 | 4 – 6 |
| 20 Newsgroups | NLP | ~1,900 docs, TF-IDF 5,000 | $2^n$ via PCA | 4 – 6 |
| Digits (8×8) | Vision | 1,797 × 64 | none needed | 6 |
| Wine | Tabular | 178 × 13 | zero-padded to 16 | 4 |

> [!WARNING]
> **Reducing E. Coli to $2^4{=}16$ genes leaves 61% of samples as all-zero vectors**, which
> any model of the form $f(x^\top P x)$ must map to one constant prediction. The effect
> disappears by 64 genes. Plain logistic regression on 256 genes reaches F1 0.812, above
> every Pauli configuration tested — the quadratic map does not add discriminative power
> on this dataset. ([ecoli_feature_count.txt](results/ecoli_feature_count.txt))

SST2 and full MNIST appear in the parent paper but are **not** implemented here: both
need sequence inputs ($s>1$, Eq. 5) and a basis beyond what the dense Pauli tensor holds.

---

## References

- **Tiblias, Schroeder, Zhang, Gachechiladze, Gurevych (2025).** *An Efficient Quantum
  Classifier Based on Hamiltonian Representations.*
  [arXiv:2504.10542](https://arxiv.org/abs/2504.10542) — the parent paper; HAM, PEFF, SIM
  and Eq. 9 are theirs.
- **Jerbi, Gyurik, Marshall, Molteni, Dunjko (2024).** *Shadows of quantum machine
  learning.* Nature Communications 15:5676 — introduces flipped models.
- **Cerezo et al. (2021).** *Variational Quantum Algorithms.* Nature Reviews Physics 3.
- **Mohammed, Z. T.** *Spectral Pauli Pruning: Efficient Feature Selection for Flipped
  Quantum Hamiltonian Classifiers* — this work
  ([pdf](pdf/Spectral_Pauli_Generator.pdf)).

See [CHANGELOG.md](CHANGELOG.md) for corrections applied to the implementation and the
manuscript claims they affect.

---

## License

MIT.
