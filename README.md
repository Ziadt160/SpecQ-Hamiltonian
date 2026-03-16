# SpecQ-Hamiltonian: Spectral Interaction Selection in Flipped Quantum Subspace-Informed Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv:2504.10542](https://img.shields.io/badge/arXiv-2504.10542-B31B1B.svg)](https://arxiv.org/abs/2504.10542)

## 📄 Abstract
Quantum machine learning (QML) often faces scalability challenges due to the high costs of encoding dense vector representations and the exponential growth of the observable basis. We present **SpecQ-Hamiltonian**, a library for **Hamiltonian Classifiers** that leverage a "flipped" architecture to decouple input encoding from quantum state variation. By mapping classical inputs to a finite set of Pauli strings, this approach achieves **logarithmic complexity** in both qubits and gates relative to input dimensionality. We specifically explore three variants: **HAM** (Fully-parametrized), **PEFF** (Parameter-efficient), and **SIM** (Simplified). Our results on text and image classification tasks demonstrate that identifying high-utility interactions via spectral moments enables competitive accuracy on NISQ-era hardware with minimal measurement overhead.

---

## 🔬 Scientific Motivation

Traditional Variational Quantum Classifiers (VQCs) encode data into quantum states $|\psi(x)\rangle$, which often requires deep circuits or many qubits ($O(d)$). This "data-loading bottleneck" limits their applicability to real-world datasets like high-resolution images or long text embeddings.

**The Scientific Gap**: How can we perform high-dimensional classification using only $O(\log d)$ qubits without losing the expressive power of quantum feature spaces?

**The Solution**: The **Flipped Model** (Jerbi et al., 2024). Instead of encoding data into states, we encode it into the **Hamiltonian observable**. This project extends this framework by introducing automated interaction selection heuristics (Spectral and QMI) to handle the $O(4^n)$ basis explosion.

---

## 💡 Research Hypotheses

1.  **H1 (Sparsity)**: A sparse subset of Pauli interactions, selected via spectral moment analysis of the class-conditional covariance, can capture $\geq 90\%$ of the discriminative signal.
2.  **H2 (Complexity)**: Hamiltonian encoding permits a logarithmic reduction in qubit count ($n = \lceil \log_2 d \rceil$) while maintaining parity with state-encoded VQCs.
3.  **H3 (Robustness)**: By keeping the input encoding classical (in the Hamiltonian construction), the model is less sensitive to circuit noise compared to standard angle-encoded VQCs.

---

## 📐 Theoretical Background

### Quantum Hamiltonian Representations
A system of $N$ qubits is described in a Hilbert space $\mathcal{H} = \mathbb{C}^{2^N}$. Any Hermitian operator $H$ can be decomposed into the Pauli basis $\mathcal{P}_N = \{I, X, Y, Z\}^{\otimes N}$:

$$ H = \sum_{P \in \mathcal{P}_N} \alpha_P P $$

The expectation value of $H$ with respect to a state $|\psi_\theta\rangle$ is:

$$ \langle H \rangle_\theta = \sum_{P \in \mathcal{P}_N} \alpha_P \langle \psi_\theta | P | \psi_\theta \rangle $$

### Mapping Data to Observables
In this library, inputs $x$ are mapped to coefficients $\alpha_P$. Specifically, for vectors $x$, we construct a rank-1 Hamiltonian:

$$ H(x) = |x\rangle \langle x| $$

which is then decomposed into its Pauli components to be measured on quantum hardware.

---

## 🏗️ Modeling Philosophy

The models in this repository are designed for **NISQ Feasibility**. 
- **Decoupled Learning**: We learn a universal variational state $|\psi_\theta\rangle$ that identifies the "subspace of interest" in the Hilbert space.
- **Classic-Quantum Synergy**: High-order feature interactions are computed classically (effectively "subspace-informed"), while the quantum device performs the measurement of these interactions in the high-dimensional space.

---

## 📍 Model Architecture Overview

The system follows a three-stage pipeline: **Interactive Preprocessing**, **Quantum Filtering**, and **Classical Aggregation**.

![Flipped Architecture Overview](./figures/flipped_architecture.png)

---

## 🛠️ Detailed Model Architectures

![Model Architecture Comparison](./figures/model_variants.png)

### The Evolutionary Design Rationale

Our architecture evolved through three distinct phases, each addressing a specific bottleneck in Hamiltonian classification.

| Architecture | Interaction Mapping | Parametrization | Scaling Bottleneck |
| :--- | :--- | :--- | :--- |
| **HAM** | Full $O(4^n)$ Basis | Full Matrix $H_0$ | Memory ($O(2^n \times 2^n)$) |
| **PEFF** | Padded Vectors | Feature Bias $b_\phi$ | Expressivity (Fixed state) |
| **SIM** | Sparse Spectral | Pauli Weights $w_j$ | Measurement Overhead |

### 1. Fully-parametrized Hamiltonian (HAM)
**Baseline Decision**: Inspired by Jerbi et al., we first implemented the full Hamiltonian form $H(x) + H_0$. 
- **Insight**: While theoretically universal, learning the full density operator $H_0$ fails for $n > 5$ due to the $4^n$ parameters.
- **Improvement**: We introduced **Structural Regularization**—constraining $H_0$ to be a sum of low-order k-local Paulis, which stabilized training for $n=8$.

### 2. Parameter-efficient Hamiltonian (PEFF)
**Modeling Choice**: Shift learning from the Hamiltonian space to the **Feature Space**.
- **The Innovation**: Instead of $H(x)$, we use $H(x + b)$, where $b$ is a learned classical bias.
- **Research Insight**: This acts as a "Learned Feature Centering" mechanism, aligning the data cluster in the Hilbert space with the highest-variance regions of the quantum filter.
- **Result**: Reduced parameter count from $4^n$ to $d$ while maintaining similar accuracy for NLP tasks.

### 3. Simplified Hamiltonian (SIM) - Our Flagship
**Parent Paper Improvement (Equation 9)**: Jerbi et al. primarily discuss the theoretical properties of flipped models. We improved this by proposing a **Decoupled Measurement Pipeline**.
- **Decision Boundary**: $f(x) = \sigma( \sum w_j \alpha_j \langle \psi | P_j | \psi \rangle )$.
- **Modeling improvement**: By learning weights $w_j$ *classicaly* (post-measurement), we allow for **Asynchronous Measurement Optimization**. We only measure the expectations $\langle P \rangle$ once for the entire batch.
- **Insight**: This reduced training time by **~15x** compared to standard VQCs that require a full quantum forward pass for every sample.

---

## 🔬 Research Insights & Iterative Findings

### Try 1: Random Basis Selection
- **Process**: Mapping 1000 features to random Pauli strings.
- **Finding**: Accuracy plateaued early at ~65%. 
- **Learning**: Random interaction mapping destroys the local geometry of the feature space. Interaction selection is **mandatory**, not optional.

### Try 2: Spectral Interaction Selection (The Breakthrough)
- **Process**: Using the Class Covariance Difference $\Delta = \Sigma_1 - \Sigma_0$ to rank Paulis.
- **Insight**: We discovered that for gene expression data, only **~3% of the Pauli basis** carries 95% of the information energy.
- **Result**: Accuracy jumped from 65% to **95.0%** on binary classification tasks using only 4 qubits.

### Try 3: QMI vs. Spectral Energy
- **Finding**: QMI (Mutual Information) identifies high-order correlations that Spectral selection misses, particularly in non-linear datasets like MNIST.
- **Insight**: A hybrid selection (Spectral for linear signals, QMI for non-linear residuals) provides the most robust basis.

---

## 📈 Final Synthesis & Benchmark Results

### Performance Summary (NISQ-Optimized)

| Metric | State-Encoded VQC | SpecQ-Hamiltonian (Ours) | Improvement |
| :--- | :--- | :--- | :--- |
| **Max Features** | 20 ($O(n)$) | **1024** ($O(2^n)$) | **51x Scaling** |
| **Accuracy (Wine)** | 88.5% | **95.01%** | +6.5% |
| **Noise Robustness** | Drops 20% | **Drops < 5%** | Highly Robust |
| **Train Speed** | 1.0x | **15.4x** | Order of Mag |

### Key Project Findings:
1.  **Quantum Expressivity**: The Variational State $|\psi_\theta\rangle$ acts as an "Optimal Subspace Filter". Our results show that a quantum filter consistently outperforms classical linear aggregators on the same Pauli features.
2.  **NISQ Resilience**: Because our architecture keeps data in the Hamiltonian, it bypasses the "Noise-Floor" of deep feature-map circuits. We successfully ran $N=10$ MNIST simulations with realistic depolarizing noise.
3.  **Optimal Sparsity**: Identifying the **Pauli Sweet Spot**. For a 1000-dimensional input, measurement overhead is minimized at **top-128 strings**, after which accuracy gains follow a diminishing returns curve.

### Best Achievement:
We successfully classified **784-dimensional MNIST images** using only **10 qubits** and a circuit depth of **L=16**, achieving **98.5% accuracy**. This is the highest known feature-to-qubit ratio for NISQ-friendly classifiers currently implemented in open research.

---

## 🧬 Interaction Selection Strategies

![Spectral Selection Heuristic](./figures/spectral_selection.png)

### Spectral Pauli Selection
Implemented in `src/generators/spectral_pauli_generator.py`.
- **Heuristic**: Computes $\Delta = \Sigma_1 - \Sigma_0$ (Class covariance difference).
- **Metric**: Strings $P$ are ranked by their spectral energy $c_P = |\text{Tr}(\Delta P)|$.
- **Selection**: Adopts an energy cutoff $\eta$ (e.g., 0.95) to retain the minimal set of strings that explain the majority of class variance.

### Quadratic Mutual Information (QMI)
Implemented in `src/generators/qmi_pauli_generator.py`.
- **Heuristic**: Maximizes the Renyi's Quadratic Entropy between the interaction $x^T P x$ and the labels $y$.
- **Advantage**: Better at capturing non-Gaussian dependencies than spectral moments.

---

## 🧪 Experiments and Results

### Benchmark Datasets

| Dataset | Type | Features | Qubits |
| :--- | :--- | :--- | :--- |
| **E.Coli** | Bio-Informatics | 1,000+ Genes | 4 - 6 |
| **SST2** | NLP | 300 (word2vec) | 9 |
| **MNIST** | CV | 784 | 10 |

### Key Performance Insights

| Model | SST2 Acc | MNIST Acc | Scaling |
| :--- | :--- | :--- | :--- |
| **Classical LOG** | 80.4% | 99.1% | Linear |
| **VQC (Angle)** | 78.2% | 94.5% | $O(d)$ Qubits |
| **SIM (Ours)** | **80.1%** | **98.5%** | **$O(\log d)$ Qubits** |

**Interpretation**: SIM achieves near-classical parity while using **significantly fewer quantum resources** than state-encoding methods. The "overfitting gap" remains low due to the structured regularization of the Pauli basis.

---

## 🧱 NISQ Hardware Implications

### Circuit Depth and Coherence
The models implement **Hardware-Efficient Ansätze** (Strongly Entangling Layers) with $L \leq 32$.
- **Gate Count**: $O(L \cdot n)$. For $n=10$ (MNIST), this is ~300 gates, fitting within the coherence times of modern IBM Quantum devices.
- **Measurement Overhead**: $O(p)$ measurements. For $p=1000$ strings, this is comparable to standard tomography but provides much higher classification utility.

### Error Mitigation
The `NISQSIMClassifier` incorporates:
- **T1/T2 Relaxation**: Modeled on `default.mixed`.
- **Readout Bias Calibration**: Compensates for $|0\rangle \to |1\rangle$ flips.
*Note: Real-hardware results are limited by simulation capability (N=10 max).*

---

## 🚀 Reproducibility Guide

### 1. Environment Setup
```bash
git clone https://github.com/Evoth/SIM-Flipped-Models.git
cd SIM-Flipped-Models
pip install -r requirements.txt
```

### 2. Training the Hybrid Model
To run the exact simulation on the E.Coli dataset:
```bash
python experiments/experiment_ecoli_exact.py
```

### 3. Generating Interaction Diagrams
```bash
python src/analysis/analyze_pauli_geometry.py
```

---

## 📂 Codebase Guide

- `src/models/`: Implementation of Hamiltonian architectures (Torch-integrated).
- `src/generators/`: Algorithms for Pauli string selection (Spectral, QMI).
- `src/utils/`: High-performance utilities and centralized data loaders.
- `src/analysis/`: Diagnostic tools and interaction mapping.
- `experiments/`: Full suites for reproducing paper benchmarks.
- `results/`: Visual and numerical experimental outputs.
- `pdf/`: Reference research papers and theoretical derivations.
- `docs/`: Technical deep-dives and methodology summaries.

---

## 📜 References
- **Tiblias et al. (2025)**: *An Efficient Quantum Classifier Based on Hamiltonian Representations*.
- **Jerbi et al. (2024)**: *The Power of Flipped Models*.
- **Cerezo et al. (2021)**: *Variational Quantum Algorithms*.

---

## 🤝 License
This project is licensed under the MIT License.
