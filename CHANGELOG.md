# Changelog

All notable changes to this project. Every correctness entry was verified numerically;
results produced before a fix should be regarded as superseded.

## [0.2.0] — 2026-10-13 — Public release

Prepared for publication as a research repository. Every results file the README cites
was regenerated with this version, so the numbers in the README are what the code
produces.

### Removed

- **Third-party material.** The PDF of Tiblias et al. (2025) and two plain-text
  extractions of it. The paper is cited by its arXiv identifier instead.
- `pdf/` (copies, or superseded copies, of figures in `results/`); `figures/`
  (AI-generated concept images, unreferenced, whose labels misnamed the model variants —
  "Quantum Heisenberg-based Model" for HAM, "Sparse Indexing Method" for SIM); the unused
  `data/trades.csv`.
- **Hydra.** `configs/`, `src/configs/`, the `outputs/` run logs, `test_hydra.py` and the
  `hydra-core`/`omegaconf` dependencies. `experiment_mnist.py` applied `@hydra.main` to a
  function that takes no config, which fails at launch.
- One-off scripts: `modify_files.py` (the codemod that introduced import-time seeding,
  below) and `experiments/fix_plot.py`.
- `experiments/results/`, a second results tree created by running scripts from inside
  `experiments/`; its files were merged into `results/`. Four case-study result files no
  current script produces.

### Fixed — library

- **Import-time reseeding.** 38 modules called `set_seed()` at import, so importing, for
  example, `src.models.sim_classifier` part-way through a run silently reset the NumPy,
  PyTorch and Python RNGs. Seeding now happens only in each script's `__main__` block,
  and a test guards against regressions.
- **Self-selecting exact models reused their first basis.** `SpectralExactSIMClassifier`
  and `FWHTExactSIMClassifier` stored the selected strings in the attribute that marks a
  user-supplied basis, so a second `fit()` kept the first fit's selection. Their
  duplicated training loop is now one base class, `BasisSelectingExactSIM`, which also
  adds `predict_proba`.
- **FWHT selection still used the second-moment Δ** after the spectral generator moved to
  the covariance. Both now share `compute_delta`, with a `moment=` argument (also added to
  `SpectralSIMClassifier`). A test checks that the two back-ends select the same strings.
- **20 Newsgroups was downloaded on every load** (14 MB over HTTP): the "already
  downloaded" check looked for an archive that scikit-learn deletes once it has built its
  cache. The cache is now used first and the upstream mirror only as a fallback.
- **`data_generator` returned labels in {−1, +1}**, unlike every other loader; spectral
  selection splits classes on `y == 0` / `y == 1` and saw an empty class. Now {0, 1}.
- **Hard-coded `d:\Evoth Labs\...` paths** in `src/utils/data_utils.py` and
  `experiment_spectral_ecoli.py`; the latter also fit χ² on test labels. Outputs are now
  written through `results_path()`, anchored at the repository root.
- `noise_utils` no longer relies on `qml.operation.Observable` (removed in PennyLane
  0.43); `NoisyVQC` rejects an unknown variant instead of failing later with
  `UnboundLocalError`; `compute_delta` warns rather than prints.

### Fixed — experiments

- **Crashes.** `experiment_noise_robustness`, `experiment_ecoli_exact_ablation` and
  `analysis_canonical_patterns` broke on the `(num_classes, p)` shape of `w`;
  `experiment_case_study_q_vs_q` imported a `VQCPennyLane` that never existed;
  `experiment_mnist` failed under Hydra.
- **Gene labels.** `analyze_pauli_geometry` labelled features selected on the training
  split with gene names from a full-data χ² selection — 12 of 16 heatmap positions named
  the wrong gene. `experiment_stability_validation` labelled every split with one gene
  list, mislabelling 10–13 of 16 positions for four of five splits.
- **Leakage.** The fraud case study uses a new `load_fraud_split`, which fits PCA and
  scaling on the training split. The single-split 20 Newsgroups benchmarks
  (`20newsgroups`, `spectral_20newsgroups`, `spectral_adaptive`, `pca_baseline`, `qda`,
  `stress_test_spectral`) use `load_20newsgroups_split`.
- `experiment_variant_comparison` counted the unused circuit's 36 weights towards
  circuit-free SIM, which has 48 parameters, not 84.
- `experiment_penalised_pruning` reported "optimal k = 4" where every k scored F1 = 0.
- `experiment_noise_robustness` damps by (1 − 4p/3)^wt(P), the exact factor for a
  per-qubit `DepolarizingChannel(p)`, putting p on the same scale as the other noise
  experiments.
- `experiment_lasso` reports the chance-level overlap alongside the measured one.
  `shot_budget` and `pruning_under_noise` now state in their reports that standardising
  after multiplying by ⟨P⟩ makes them blind to noise by construction.
- `experiment_nisq_sweep` writes a CSV; bare `except:` clauses narrowed to `ValueError`;
  `experiment_monte_carlo_validity` and the `experiment_mnist` split are seeded; plot
  titles no longer assert conclusions or call the 8×8 Digits dataset "MNIST".

### Documentation

- README: removed the dead link to an unpublished manuscript PDF and the arXiv badge;
  fixed installation (Python ≥ 3.10) and data provenance; added a citation section.
  Corrected claims:
  - the Lasso "corroboration" is at chance level (86.4% overlap against 86.7% for a
    random subset) and is no longer presented as evidence;
  - two of the three noise experiments cannot detect noise by construction, and the third
    *does* show degradation under strong regularisation — the README had said none did;
  - PEFF uses a third fewer parameters than SIM, not a third as many;
  - the 20 Newsgroups projection is fit transductively (label-free) in the main
    experiments; the README had said all preprocessing was train-only.
- `docs/methodology.md` rewritten: it described the superseded second-moment Δ and a
  noise-robustness finding the results contradict. `docs/experiments.md` catalogues every
  script with its data, outputs and runtime.

### Added

- `tests/` — 41 tests of the claims the results rest on, run in CI (ruff + pytest on
  Python 3.10 and 3.12).
- `LICENSE` (the README declared MIT but no licence file existed), `CITATION.cff`,
  `.gitattributes`, `src/utils/paths.py`.
- `requires-python >= 3.10` (the pinned dependencies already required it) and
  `pennylane < 0.43`.

### Results

Most committed results predated either the switch of Δ's default to the covariance or
the fixes above. Regenerated with this version:

- **Headline (README §5–7):** `pruning_limits`, `measurement_efficiency`,
  `spectral_vs_random_live`, `lasso`, `circuit_ablation`, `variant_comparison`,
  `shot_budget`, `pruning_under_noise`, `penalised_pruning`, `delta_definition`,
  `ecoli_feature_count`, `sweep_and_normalization`. The headline figures are unchanged —
  24 strings for 95% of full-basis F1 (86.7×), the 48/52% dead-filter/ranking split,
  17/20 wins. What moved:
  - circuit ablation: trained / frozen / removed = 0.9394 / 0.9377 / 0.9387, all within
    one standard deviation (was 0.9363 / 0.9373 / 0.9346 with the second-moment basis);
  - variant comparison: SIM 0.9317 / 0.8207 and circuit-free SIM 0.9329 / 0.8231
    accuracy on E. Coli / 20 News — circuit-free SIM is now best on both;
  - every dead string ranks after every live one on both datasets (the 20 News $n=6$
    exception was an artefact of the second-moment Δ, under which the identity string's
    coefficient is exactly zero for row-normalised inputs);
  - Lasso overlap 86.4% (was 88.98%), against 86.7% chance.
- **Benchmarks:** `20newsgroups`, `spectral_20newsgroups`, `spectral_ecoli`,
  `spectral_adaptive`, `pca_baseline`, `qda`, `mnist`, `monte_carlo_validity`,
  `noise_robustness`, `ecoli_exact_ablation`, `stability_validation`, `universal_pauli`,
  and the three `src/analysis` scripts. The gene-interaction and stability reports now
  name the correct genes.
- **Fraud case study:** all six scripts, with the leak-free loader.

Not regenerated, because of their runtime and because the reviewed changes leave their
outputs unaffected: `overfitting_gap`, `small_data`, `feature_noise`, `nisq_sweep` (so its
new CSV is not yet committed), `pruning_comparison`, `wine`, `wine_full`,
`base_experiment` and `generative_poc`. Those that select strings spectrally may predate
the covariance default.

## Known limitations

- Experiments run at $n \in \{4, 6\}$; the parent work operates at $n \in \{9, 10\}$.
- The main 20 Newsgroups experiments fit TF-IDF/PCA on the whole corpus (label-free);
  the Digits and Wine loaders standardise before splitting, and
  `experiment_universal_pauli` fits PCA on all of Digits.
- `experiment_generative_poc` is not a working generative model: its circuit never
  receives the input features, so its output does not depend on context.
- `experiment_universal_pauli` uses a hard-coded top-32 string list taken from an earlier
  E. Coli ranking.
- `experiment_pruning_limits` compares spectral and random overfitting at the full basis,
  where both arms hold the same strings; the comparison needs an intermediate $k$.
- `experiment_overfitting_gap` draws its random baseline from the full basis, dead
  strings included, and seeds it with `seed + k`, which collides across seeds.
- `experiment_monte_carlo_validity` validates the second-moment estimator, not the
  default covariance Δ.
- Several benchmark scripts use a single split or seed, and some report accuracy alone
  on the 13.7%-positive E. Coli labels. `overfitting_gap`, `small_data`, `feature_noise`
  and `noise_robustness` write only figures; `advanced_baselines` prints to stdout.
- `experiment_qpu_test`'s `qiskit.ibmq` path targets a PennyLane-Qiskit device since
  replaced upstream by `qiskit.remote`, and takes the API token on the command line.

## [0.1.1] — 2026-07-28 — Verification and repair

Corrections applied to the implementation, and the manuscript claims they affect.


### Correctness

#### Selection

- **`Δ` was the second moment, not the covariance.** `Δ = E[xxᵀ|y=1] − E[xxᵀ|y=0]` with no
  mean subtraction, while Eq. 4 specifies covariance. These differ whenever the
  class-conditional means are non-zero, which is always the case here — the two select
  bases with only **21–25% overlap at k=16**, converging to 94% by k=128. Now selectable
  via `moment=`, defaulting to `'covariance'`.

  A controlled 5-seed comparison found them **statistically indistinguishable on
  accuracy** (+0.040 at k=8, +0.074 at k=16, −0.041 at k=128, against per-seed standard
  deviations of 0.02–0.08). The default is chosen on principle, not measured performance.

- **QMI ranked features in reverse.** `qmi_score` returned `-log(V_xy²/(V_x·V_y))`, which
  *decreases* with dependence. On a controlled test it scored an independent feature
  **1.6268**, a partially informative one 1.2243, and a perfect predictor **0.2708** —
  then sorted descending, so `generate_qmi_pauli_strings` selected the *least* informative
  strings. Replaced with the Cauchy-Schwarz QMI, `log(V_J·V_M/V_C²)`, which is ≥0 and zero
  exactly at independence. Now orders correctly: perfect 0.4005 > useful 0.2799 > weak
  0.0381 > independent 0.0128 > constant 0.0000. **Any prior QMI result is invalid.**

- **`NameError` on the degenerate-class path.** The `len(X0) < 2` guard assigned
  `cov0`/`cov1` while the following line read `R1 - R0`. Also guarded division by zero when
  `Δ` is identically zero.

#### Synthetic data

- **The `conditional` regime was 75% contaminated.** The filter required one `X` and one
  `Z` but omitted the `Y` exclusion, so 36 of 48 terms (e.g. `XZYY`) were not `ZᵢXⱼ`
  interactions at all. After the fix the diagnostic behaves as designed — each regime is
  solved exactly when its matching Pauli class enters the basis:

  | regime | before matching class | after | class |
  | :--- | ---: | ---: | :--- |
  | linear | 0.535 | 0.995 | C1Z |
  | pairwise | 0.495 | 0.990 | C2XX |
  | conditional | 0.575 | **0.985** (was 0.625 max) | C2ZX |

#### Noise model

Five defects in `NISQSIMClassifier`. With all noise zeroed it now reproduces
`ExactSIMClassifier` exactly (max Δ⟨P⟩ = 3.3e-16); it did not before.

- **Wrong ansatz.** A range-1 CNOT ring was hard-coded in every layer;
  `StronglyEntanglingLayers` cycles ranges `[1,2,3]`. The two circuits differed by up to
  **0.56** in ⟨P⟩ *with noise switched off*, so Exact-vs-NISQ comparisons were measuring an
  ansatz change rather than noise.
- **`PhaseDamping` parametrised with T2**, double-counting the dephasing that amplitude
  damping already causes. T1=50 ms, T2=70 ms realised **T2_eff = 58.3 ms**. Correct
  construction uses pure dephasing, `1/T_φ = 1/T2 − 1/(2·T1)`, `γ = 1−exp(−2t/T_φ)`.
  T2 > 2·T1 is now rejected as unphysical.
- **Thermal noise was numerically inert.** `gate_time`=100 ns against T1=50 ms gives
  p_amp ≈ 2e-6, **5,000× below** the two-qubit depolarizing rate. 50 ms is a trapped-ion
  coherence time; defaults are now superconducting (T1=100 µs, T2=80 µs, 200 ns), a 5×
  ratio.
- **Readout error did not affect X observables.** Modelled as a `BitFlip` channel before
  measurement — but X commutes with the flip, so ⟨X⟩ was left *exactly* unchanged while
  ⟨Y⟩, ⟨Z⟩ scaled by (1−2p). Measured: ⟨XX⟩ ratio 1.0000 vs ⟨ZZ⟩ 0.9216. Now attenuates by
  (1−2p)^weight, basis-independently.
- **Silent complex→float cast** on the Pauli stack, now explicit via `real_pauli_stack`.

Separately, `experiment_noise_robustness.py` damps by `(1−p)^weight`. The correct factor
for per-qubit `DepolarizingChannel(p)` is **(1−4p/3)^weight** — verified against a
density-matrix simulation to 1.6e-13. At p=0.2 that is 0.733 vs 0.800, so the figure
systematically under-damps.

#### Data handling

- **Test-label leakage in the E. Coli loader.** `SelectKBest(chi2)` was fit on the full
  dataset before splitting. Added `load_ecoli_raw` / `select_topk_chi2` /
  `load_ecoli_split`, which fit selection on the training split only, and migrated all
  eight evaluation call sites. A third inlined copy of the leaky loader inside
  `experiment_pruning_comparison.py` was removed.

  Impact was smaller than expected: train-only and full-data selection agree on **15 of 16
  genes**, and the leak-free E. Coli numbers (0.9466 / 0.9449) are within noise of the
  originals (0.9432 / 0.9501). The fix is still correct and retained.

  `load_ecoli_reduced` is kept for non-evaluative geometry analysis and warns at call
  time. `load_20newsgroups_split` added as the leak-free text equivalent.

### Model fidelity

- **HAM and PEFF were advertised but absent.** Both now implemented in
  `hamiltonian_variants.py`, evaluating the full Hamiltonian expectation from the state
  vector rather than a Pauli sum. Measured parameter counts reproduce the paper's scaling:
  at n=4 with two ansatz layers, HAM = 280 (O(4ⁿ)), PEFF = 42 (O(d)); 292 and 54 with three.

  Note PEFF's Hamiltonian is positive semi-definite as written, so ψ†Hψ ≥ 0 always and a
  fixed 0.5 threshold emits one class for every input. A single learnable scalar offset was
  added to make it trainable; the paper's equations omit this.

- Added the **1/2ⁿ prefactor** of Eq. 7/9 (verified equivalent — rescaling `w` by 2ⁿ gives
  identical outputs), **multi-class support** via `c` weight vectors sharing one
  measurement pass (Sec. 3.3), and **sequence inputs** `(batch, s, dim)` for the Eq. 5 mean.

### Performance

- **Spectral projection.** Replaced a loop of 4ⁿ `np.trace(Δ @ P)` products, O(4ⁿ·2³ⁿ),
  with a single `einsum('ij,kji->k')` at O(4ⁿ·2²ⁿ) — **5.8× faster measured at n=4**,
  widening as 2ⁿ. Values agree to 3.5e-18; the first ranking change is at position 136,
  exactly where the identically-zero strings begin, so live-string ordering is unchanged.

  Note this is **not** an eigendecomposition. §VI.B of the manuscript describes selection
  as one at O(2³ᴺ); no such routine exists, and `scipy.linalg` was imported but never
  called.

### Packaging and structure

- **37 unresolved imports across 20 files**, plus 8 names used but never bound. The
  package had been reorganised into `src/{models,utils,generators,analysis}/` without
  updating imports; `sim_classifier.py` imported `.pauli_utils` from the wrong package,
  taking the base model class down with it. Added the five missing `__init__.py` files.
- **`experiment_mnist.py` and `experiment_wine.py` redefined their loaders** immediately
  after importing them, shadowing the import, and the local copies referenced names the
  files never imported. Both duplicates removed; the canonical loaders are equivalent.
  Same pattern in `analysis_canonical_patterns.py`.
- **`pyproject.toml` was untracked**, so a fresh clone could not import anything. Now
  committed, with `hydra-core`, `omegaconf` and `tqdm` removed (imported nowhere) and
  `scipy` and `seaborn` added (required but undeclared).
- Hardcoded absolute `d:\Evoth Labs\...` paths replaced with package-relative resolution;
  output paths normalised so every experiment writes to the repository-root `results/`.

### Manuscript corrections

Pending in *Spectral Pauli Pruning*, all verified against the current code:

| § | Claim | Correction |
| :--- | :--- | :--- |
| III.B Eq. 4 | Δ is a covariance difference | code computed second moments; now aligned |
| VI.B | selection is an eigendecomposition at O(2³ᴺ) | no eigendecomposition exists; cost is O(4ⁿ·2²ⁿ) after the einsum fix |
| Abstract | "83.49% with only **57.6%** of the full Pauli basis" | k=1737 of 4096 is **42.4%**; 57.6% is the compression |
| V.B | Exact SIM 84.23% | committed run gives 0.8330 |
| V.B | ⟨P⟩_θ provides a "non-trivial inductive bias" | that comparison confounds optimizer, scaling and bias term; controlled ablation finds no expressivity contribution |

