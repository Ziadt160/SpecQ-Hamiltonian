# Experiment catalogue

Every script in [`experiments/`](../experiments) runs from the repository root as a module
and writes to [`results/`](../results):

```bash
python -m experiments.experiment_pruning_limits
```

Runtimes are rough single-machine CPU figures. Scripts that use 20 Newsgroups download it
on first run (scikit-learn caches it afterwards); the fraud case study needs
`data/creditcard_2023.csv` (see the README, "Data"). Unless noted, splits are 70/30 with a
fixed `random_state`, and everything that sees labels is fit on the training split.

## Headline results (README §5–7)

| Script | Question | Data | Outputs | Runtime |
| :--- | :--- | :--- | :--- | :--- |
| [`pruning_limits`](../experiments/experiment_pruning_limits.py) | Smallest $k$ reaching 90/95/99% of full-basis F1; train/test gap vs $k$ | E. Coli, 20 News; $n=4,6$; 5 seeds | `pruning_limits.{csv,txt,png}` | 10–30 min |
| [`measurement_efficiency`](../experiments/experiment_measurement_efficiency.py) | How many strings are identically zero, and where they fall in random draws and spectral rankings | synthetic; E. Coli, 20 News $n=4,6$ | `measurement_efficiency{,_draws}.csv`, `.txt`, `.png` | 1–2 min |
| [`spectral_vs_random_live`](../experiments/experiment_spectral_vs_random_live.py) | Splits the spectral advantage into dead-string filtering and genuine ranking | E. Coli, 20 News $n=4,6$; 5 seeds × 5 draws | `spectral_vs_random_live.{csv,txt,png}` | 5–15 min |
| [`lasso`](../experiments/experiment_lasso.py) | Overlap of L1-logistic selection with the spectral top-$k$, against chance | 20 News $n=4$ | `lasso_comparison.txt` | < 1 min |
| [`circuit_ablation`](../experiments/experiment_circuit_ablation.py) | Circuit trained vs frozen at init vs removed | E. Coli $n=4$, spectral top-32; 5 init seeds | `circuit_ablation.{csv,txt,png}` | a few min |
| [`variant_comparison`](../experiments/experiment_variant_comparison.py) | HAM vs PEFF vs SIM vs circuit-free SIM | E. Coli, 20 News $n=4$; 3 seeds | `variant_comparison.{csv,txt,png}` | 2–5 min |
| [`shot_budget`](../experiments/experiment_shot_budget.py) | Finite-shot estimates of $\langle P\rangle$ for spectral vs random bases | E. Coli, 20 News $n=4$; 3 seeds | `shot_budget.{csv,txt,png}` | 2–5 min |
| [`pruning_under_noise`](../experiments/experiment_pruning_under_noise.py) | Optimal $k$ vs depolarising rate and shot budget | E. Coli, 20 News $n=4$; 3 seeds | `pruning_under_noise.{csv,txt,png}` | 2–5 min |
| [`penalised_pruning`](../experiments/experiment_penalised_pruning.py) | Optimal $k$ under noise when $w$ is L2-penalised and scaling is fixed | E. Coli, 20 News $n=4$; 3 seeds | `penalised_pruning.{csv,txt,png}` | 1–3 min |
| [`delta_definition`](../experiments/experiment_delta_definition.py) | Covariance vs second-moment $\Delta$: top-$k$ overlap and F1 | E. Coli, 20 News $n=4,6$; 5 seeds | `delta_definition.{csv,txt,png}` | 3–10 min |
| [`ecoli_feature_count`](../experiments/experiment_ecoli_feature_count.py) | Blank-row fraction and LR/SIM performance vs number of $\chi^2$ genes (16–2048) | E. Coli, one split | `ecoli_feature_count.{csv,txt,png}` | 1–3 min |
| [`sweep_and_normalization`](../experiments/experiment_sweep_and_normalization.py) | F1 vs basis size as a share of the live basis; L2 row-normalisation ablation | E. Coli $n=4,6$; E. Coli, 20 News $n=4$; 3 seeds | `pauli_count_sweep.csv`, `normalization_ablation.csv`, `sweep_and_normalization.{txt,png}` | 5–10 min |

Two caveats apply to the noise rows. `shot_budget` and `pruning_under_noise` standardise
features after multiplying by the estimated $\langle P_j\rangle$, which divides out any
per-string damping or shot error, so they cannot detect noise by construction (each
script says so in its report). `penalised_pruning` removes that escape.

<!-- BENCHMARKS -->

## Noise, hardware and baselines

| Script | Question | Data | Outputs | Runtime |
| :--- | :--- | :--- | :--- | :--- |
| [`noise_robustness`](../experiments/experiment_noise_robustness.py) | Test-time damping $(1-4p/3)^{\mathrm{wt}(P)}$ of a clean-trained SIM: full basis vs spectral top-50 | 20 News $n=4$; 5 seeds | `noise_robustness.png` | 5–15 min |
| [`nisq_sweep`](../experiments/experiment_nisq_sweep.py) | Spectral vs random basis under `NISQSIMClassifier` (gate, $T_1/T_2$, readout noise) | E. Coli $n=4$; 3 seeds × 150 epochs | `nisq_robustness.{csv,png}` | 15–45 min |
| [`feature_noise`](../experiments/experiment_feature_noise.py) | Gaussian noise on training features; spectral top-32 (reselected) vs full basis | E. Coli $n=4$; 5 seeds | `feature_noise_sweep.png` | 15–30 min |
| [`pruning_comparison`](../experiments/experiment_pruning_comparison.py) | Spectral vs QMI vs random ranking at $k \in \{32, 50, 128\}$ | 20 News, E. Coli $n=4$; one split | `pruning_ablation_results.csv`, `pruning_comparison_*.png` | 3–10 min |
| [`advanced_baselines`](../experiments/experiment_advanced_baselines.py) | Spectral vs locality (lowest weight) vs random at $k=50$ | 20 News $n=4$ | stdout only | 1–2 min |
| [`generative_poc`](../experiments/experiment_generative_poc.py) | Toy character-level "generative flipped model" — **not functional**: the circuit ignores its input features, so the output does not depend on context | one hard-coded sentence | `generative_poc_loss.png` | 1–3 min |

## Case study: credit-card fraud

Requires `data/creditcard_2023.csv`. All scripts use `load_fraud_split`, which fits PCA and
scaling on the training split only, and draw a class-balanced sample from the 568k rows.
`case_study_noise` and `topk_noise_sweep` resume from their CSVs; delete those to
regenerate from scratch.

| Script | Question | Sample | Outputs (in `results/case_study_fraud/`) | Runtime |
| :--- | :--- | :--- | :--- | :--- |
| [`case_study_q_vs_c`](../experiments/experiment_case_study_q_vs_c.py) | `SpectralSIMClassifier` ($k=64$, $n=6$) vs RF / GBM / MLP / LR | 5,000 | `q_vs_c_{metrics.csv,performance.png}` | < 1 min |
| [`case_study_q_vs_q`](../experiments/experiment_case_study_q_vs_q.py) | Spectral SIM ($k=25$) vs two VQCs vs a fidelity-kernel QSVM | 1,000 | `q_vs_q_{metrics.csv,performance.png}` | 10–20 min |
| [`case_study_analysis`](../experiments/experiment_case_study_analysis.py) | FWHT-selected SIM (top-15) vs LR / RF / SVM, with a Pauli-string report | 1,000 | `spectral_deductions_comparative.txt`, `models_performance_comparison.png`, `spectral_importance_profile.png` | ~2 min |
| [`case_study_noise`](../experiments/experiment_case_study_noise.py) | Spectral SIM, two VQCs and QSVM under depolarising noise $p \le 0.2$ | 500 | `noise_results_comprehensive.csv`, `noise_overfitting_analysis.png`, `noise_test_auc_comparison.png` | ~1 h |
| [`topk_noise_sweep`](../experiments/experiment_topk_noise_sweep.py) | Test AUC/F1 for $k \in \{4,\dots,32\}$ × $p$, trained clean, noisy at evaluation | 500 | `topk_noise_results.csv`, `topk_noise_resilience{,_f1}.png` | 2–5 min |
| [`qpu_test`](../experiments/experiment_qpu_test.py) | Spectral SIM (top-16) with finite shots on a PennyLane device; `default.qubit` unless told otherwise | 100 | `qpu_test_results.csv` | ~20 s on the simulator |

`qpu_test` never contacts hardware by default. Its `--device qiskit.ibmq` path targets a
PennyLane-Qiskit device that has since been replaced upstream by `qiskit.remote`.
