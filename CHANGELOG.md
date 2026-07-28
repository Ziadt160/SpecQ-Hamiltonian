# Changelog

Corrections applied to the implementation, and the manuscript claims they affect.
Every entry was verified numerically; results predating a fix should be regarded as
superseded.

## Correctness

### Selection

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

### Synthetic data

- **The `conditional` regime was 75% contaminated.** The filter required one `X` and one
  `Z` but omitted the `Y` exclusion, so 36 of 48 terms (e.g. `XZYY`) were not `ZᵢXⱼ`
  interactions at all. After the fix the diagnostic behaves as designed — each regime is
  solved exactly when its matching Pauli class enters the basis:

  | regime | before matching class | after | class |
  | :--- | ---: | ---: | :--- |
  | linear | 0.535 | 0.995 | C1Z |
  | pairwise | 0.495 | 0.990 | C2XX |
  | conditional | 0.575 | **0.985** (was 0.625 max) | C2ZX |

### Noise model

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

### Data handling

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

## Model fidelity

- **HAM and PEFF were advertised but absent.** Both now implemented in
  `hamiltonian_variants.py`, evaluating the full Hamiltonian expectation from the state
  vector rather than a Pauli sum. Measured parameter counts reproduce the paper's scaling:
  at n=4, HAM = 280 (O(4ⁿ)), PEFF = 42 (O(d)).

  Note PEFF's Hamiltonian is positive semi-definite as written, so ψ†Hψ ≥ 0 always and a
  fixed 0.5 threshold emits one class for every input. A single learnable scalar offset was
  added to make it trainable; the paper's equations omit this.

- Added the **1/2ⁿ prefactor** of Eq. 7/9 (verified equivalent — rescaling `w` by 2ⁿ gives
  identical outputs), **multi-class support** via `c` weight vectors sharing one
  measurement pass (Sec. 3.3), and **sequence inputs** `(batch, s, dim)` for the Eq. 5 mean.

## Performance

- **Spectral projection.** Replaced a loop of 4ⁿ `np.trace(Δ @ P)` products, O(4ⁿ·2³ⁿ),
  with a single `einsum('ij,kji->k')` at O(4ⁿ·2²ⁿ) — **5.8× faster measured at n=4**,
  widening as 2ⁿ. Values agree to 3.5e-18; the first ranking change is at position 136,
  exactly where the identically-zero strings begin, so live-string ordering is unchanged.

  Note this is **not** an eigendecomposition. §VI.B of the manuscript describes selection
  as one at O(2³ᴺ); no such routine exists, and `scipy.linalg` was imported but never
  called.

## Packaging and structure

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

## Manuscript corrections

Pending in *Spectral Pauli Pruning*, all verified against the current code:

| § | Claim | Correction |
| :--- | :--- | :--- |
| III.B Eq. 4 | Δ is a covariance difference | code computed second moments; now aligned |
| VI.B | selection is an eigendecomposition at O(2³ᴺ) | no eigendecomposition exists; cost is O(4ⁿ·2²ⁿ) after the einsum fix |
| Abstract | "83.49% with only **57.6%** of the full Pauli basis" | k=1737 of 4096 is **42.4%**; 57.6% is the compression |
| V.B | Exact SIM 84.23% | committed run gives 0.8330 |
| V.B | ⟨P⟩_θ provides a "non-trivial inductive bias" | that comparison confounds optimizer, scaling and bias term; controlled ablation finds no expressivity contribution |

## Known limitations

- `experiment_nisq_sweep.py` writes only a PNG — no `.txt` or `.csv`, unlike every other
  experiment, so its numbers are not machine-readable.
- `experiment_pruning_limits.py` compares spectral vs random overfitting *at the full
  basis*, where both arms contain the same strings; that comparison is degenerate and
  needs an intermediate k.
- Two bare `except:` clauses remain in `experiment_advanced_baselines.py` and
  `experiment_pruning_comparison.py`, wrapping metric computation.
- `data_generator` returns labels in {−1,+1} while every other loader uses {0,1}. sklearn
  tolerates this; `BCELoss` would not.
- `analysis_canonical_patterns` and `stress_test_spectral` use a single fixed seed, so
  their numbers carry no error bars.
