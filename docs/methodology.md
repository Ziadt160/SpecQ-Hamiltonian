# Methodology: Simplified Hamiltonian (SIM) Classifiers

## 1. Interaction-Aware Feature Mapping

The core principle of the SIM classifier is to leverage the non-linear relationship between classical features via Pauli interactions. For a vector $x \in \mathbb{R}^D$, we define the interaction feature under operator $P$ as:
$$ \phi_P(x) = x^T P x $$
where $P$ is a $D \times D$ Pauli tensor product.

## 2. Spectral Pauli Selection

To avoid the exponential explosion of the Pauli basis, we employ **Spectral Moment Selection**. We compute the difference in class-conditional second moments:
$$ \Delta = \mathbb{E}[xx^T | y=1] - \mathbb{E}[xx^T | y=0] $$
The importance of a Pauli string $P$ is given by the magnitude of its projection onto $\Delta$:
$$ c_P = \frac{1}{2^N} \text{Tr}(\Delta P) $$

## 3. Flipped Architecture

The "Flipped" architecture (implemented in `ExactSIMClassifier`) decouples the input encoding from the variational state.
- **Classical Branch**: Computes quadratic forms $x^T P_j x$.
- **Quantum Branch**: Pre-computes expectation values $E_j = \langle \psi_\theta | P_j | \psi_\theta \rangle$.
- **Combination**: The final logit is the weighted inner product of classical features and quantum expectations:
$$ z = \sum_j (x^T P_j x) \cdot w_j \cdot E_j $$

## 4. Optimization Strategy

We use **Stochastic Gradient Descent (Adam)** to jointly optimize:
1. **Circuit Parameters ($\theta$)**: Learns the optimal state $|\psi_\theta\rangle$ for interaction filtering.
2. **Pauli Weights ($w$)**: Corrects for global importance of specific strings.
3. **Input Bias ($b$)**: Shifts the input distribution to align with the Pauli basis.
