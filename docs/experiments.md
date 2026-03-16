# Experimental Analysis

This document summarizes the experiments conducted within this repository to validate the SIM-Flipped hypothesis.

## 1. Benchmarking on Biological Data (E.Coli)
- **Script**: `experiments/experiment_ecoli_exact.py`
- **Objective**: Validate performance on high-dimensional gene expression data.
- **Method**: Select top 16 genes via Chi-squared selection, map to 4 qubits.
- **Finding**: Achieving >90% accuracy with a reduced Pauli set (top 32 strings).

## 2. Noise Robustness (NISQ Sweep)
- **Script**: `experiments/experiment_nisq_sweep.py`
- **Objective**: Analyze the impact of thermal relaxation and depolarizing noise.
- **Method**: Sweeping $T_1/T_2$ times and gate error rates in `NISQSIMClassifier`.
- **Finding**: The hybrid architecture maintains higher accuracy than pure VQCs at moderate noise levels because the input encoding is classical and exact.

## 3. Generalization and Overfitting
- **Script**: `experiments/experiment_overfitting_gap.py`
- **Objective**: Measure the gap between training and test performance.
- **Method**: Training on small subsets of the Wine dataset.
- **Finding**: SIM models show strong regularization properties when weights $w$ are penalized (L2 regularization).

## 4. Canonical Pattern Analysis
- **Script**: `src/analysis/canonical_patterns.py`
- **Objective**: Identify the distribution of "winning" Pauli strings.
- **Metrics**: Counts of $C_{1Z}$, $C_{1X}$, $C_{2ZX}$ etc.
- **Finding**: Diagonal strings ($Z$-only) dominate initial learning, but off-diagonal strings ($X$-containing) are critical for capturing cross-feature correlations.
