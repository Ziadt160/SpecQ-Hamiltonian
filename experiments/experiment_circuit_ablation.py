"""
Does the variational quantum state contribute anything to the SIM classifier?

The shipped decision function (Eq. 9) is

    f(x) = sigma( sum_j (x^T P_j x) * w_j * <psi_theta| P_j |psi_theta> )

The expectation e_j = <psi_theta| P_j |psi_theta> does not depend on x -- it is
recomputed once per forward pass, not once per sample. The weight w_j is a free
unconstrained parameter of the same shape. So the composite coefficient

    v_j = w_j * e_j

reaches any value in R^K whenever e_j != 0 (take w_j = v_j / e_j). The
hypothesis class is therefore *identical* to logistic regression on the
classical quadratic features phi_j(x) = x^T P_j x, and the circuit should be
removable with no loss of expressivity.

This experiment tests that directly with three arms:

    trained  circuit weights and w both optimized  (the shipped model)
    frozen   circuit weights fixed at random init, only w optimized
    none     circuit removed entirely (e_j := 1), only w optimized

If the circuit supplies expressivity, `trained` should beat `frozen` and both
should beat `none`. If it is a redundant reparametrization, all three land in
the same place -- and `none` may in fact do better, because the multiplicative
e_j factors rescale gradients unevenly and slow the optimizer.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.utils.data_loader import load_ecoli_split
from src.generators.spectral_pauli_generator import generate_spectral_pauli_strings
from src.models.exact_sim_classifier import ExactSIMClassifier

ARMS = ['trained', 'frozen', 'none']


class AblatedSIMClassifier(ExactSIMClassifier):
    """ExactSIMClassifier with the quantum branch optionally frozen or removed."""

    def __init__(self, n_qubits, n_layers=3, pauli_strings=None, mode='trained'):
        super().__init__(n_qubits, n_layers, pauli_strings=pauli_strings)
        if mode not in ARMS:
            raise ValueError(f"mode must be one of {ARMS}, got {mode!r}")
        self.mode = mode
        if mode == 'frozen':
            # Keep the random init, but stop optimizing it.
            self.circuit_weights.requires_grad_(False)
        elif mode == 'none':
            # The circuit is not used at all; drop it from the parameter set.
            self.circuit_weights.requires_grad_(False)

    def forward(self, x):
        x_tilde = x + self.b
        feats = torch.einsum('bm,kmn,bn->bk', x_tilde, self.P_tensor, x_tilde)

        if self.mode == 'none':
            coeffs = self.w
        else:
            e = self.qnode(self.circuit_weights)
            if isinstance(e, (list, tuple)):
                e = torch.stack(e)
            coeffs = self.w * e

        return torch.sigmoid(torch.sum(feats * coeffs, dim=1))


def train_and_eval(mode, pauli_strings, X_train, y_train, X_test, y_test,
                   n_qubits=4, seed=0, epochs=150, lr=0.01):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AblatedSIMClassifier(n_qubits, n_layers=3,
                                 pauli_strings=pauli_strings, mode=mode)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr)
    crit = nn.BCELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    X_te = torch.tensor(X_test, dtype=torch.float64)

    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(X_tr), y_tr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        probs = model(X_te).numpy()
    preds = (probs > 0.5).astype(int)

    return {
        'accuracy': accuracy_score(y_test, preds),
        'balanced_accuracy': balanced_accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds, zero_division=0),
        'train_loss': loss.item(),
    }


def run_ablation(n_seeds=5, top_k=32, epochs=150):
    os.makedirs('results', exist_ok=True)

    X_train, X_test, y_train, y_test = load_ecoli_split(n_qubits=4, test_size=0.3,
                                                        random_state=42)
    majority = max(y_test.mean(), 1 - y_test.mean())
    print(f"E. Coli N=4: train {X_train.shape}, test {X_test.shape}")
    print(f"Majority-class baseline on test: {majority:.4f}")

    # Spectral basis, ranked on the training split only.
    ranking, _, _ = generate_spectral_pauli_strings(X_train, y_train, 4)
    basis = ranking[:top_k]
    print(f"Spectral top-{top_k} basis selected from {len(ranking)} strings.\n")

    rows = []
    for seed in range(n_seeds):
        print(f"--- seed {seed + 1}/{n_seeds} ---", flush=True)
        for mode in ARMS:
            m = train_and_eval(mode, basis, X_train, y_train, X_test, y_test,
                               seed=seed, epochs=epochs)
            m.update(seed=seed, mode=mode)
            rows.append(m)
            print(f"  {mode:8s} acc={m['accuracy']:.4f} "
                  f"bal={m['balanced_accuracy']:.4f} f1={m['f1']:.4f} "
                  f"loss={m['train_loss']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('results/circuit_ablation.csv', index=False)

    summary = df.groupby('mode')[['accuracy', 'balanced_accuracy', 'f1', 'train_loss']]
    mean, std = summary.mean(), summary.std()

    lines = [
        "Circuit Ablation: does the variational state add expressivity?",
        "=" * 62,
        f"Dataset: E. Coli (CTZ), N=4, spectral top-{top_k} basis",
        f"Seeds: {n_seeds}, epochs: {epochs}",
        f"Majority-class baseline (test): {majority:.4f}",
        "",
        f"{'arm':<10}{'accuracy':>18}{'bal-acc':>12}{'F1':>10}{'train loss':>13}",
    ]
    for mode in ARMS:
        lines.append(
            f"{mode:<10}{mean.loc[mode,'accuracy']:>10.4f} +-{std.loc[mode,'accuracy']:<6.4f}"
            f"{mean.loc[mode,'balanced_accuracy']:>12.4f}{mean.loc[mode,'f1']:>10.4f}"
            f"{mean.loc[mode,'train_loss']:>13.4f}"
        )
    lines += [
        "",
        "trained = circuit optimized (shipped model)",
        "frozen  = circuit fixed at random init, only w optimized",
        "none    = circuit removed entirely (e_j = 1), only w optimized",
        "",
        "If the three arms agree, the quantum branch is a redundant",
        "reparametrization of a logistic regression on x^T P_j x.",
    ]
    report = "\n".join(lines)
    print("\n" + report)

    with open('results/circuit_ablation.txt', 'w') as f:
        f.write(report + "\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(ARMS))

    ax = axes[0]
    ax.bar(x, [mean.loc[m, 'accuracy'] for m in ARMS],
           yerr=[std.loc[m, 'accuracy'] for m in ARMS],
           capsize=6, color=['#4c72b0', '#dd8452', '#55a868'])
    ax.axhline(majority, ls='--', color='grey', label=f'majority class ({majority:.3f})')
    ax.set_xticks(x); ax.set_xticklabels(ARMS)
    ax.set_ylabel('Test accuracy')
    ax.set_ylim(0.80, 1.0)
    ax.set_title(f'Accuracy by circuit arm (mean +- sd, {n_seeds} seeds)')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1]
    ax.bar(x, [mean.loc[m, 'train_loss'] for m in ARMS],
           yerr=[std.loc[m, 'train_loss'] for m in ARMS],
           capsize=6, color=['#4c72b0', '#dd8452', '#55a868'])
    ax.set_xticks(x); ax.set_xticklabels(ARMS)
    ax.set_ylabel('Final training loss (BCE)')
    ax.set_title('Optimization quality at fixed epoch budget')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/circuit_ablation.png', dpi=150)
    print("\nSaved results/circuit_ablation.{csv,txt,png}")


if __name__ == "__main__":
    run_ablation()
