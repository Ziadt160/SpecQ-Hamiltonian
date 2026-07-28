"""
HAM vs PEFF vs SIM -- the first head-to-head of all three paper variants.

The README advertised three architectures, but only SIM existed in code; HAM
(Eq. 2-3) and PEFF (Sec. 3.2) are implemented in src/models/hamiltonian_
variants.py. This experiment runs all three under identical data, optimizer and
seed control, and reports the parameter counts that motivate the paper's
progression HAM -> PEFF -> SIM:

    HAM   O(4^n) parameters   full Hamiltonian expectation, matrix bias H^0
    PEFF  O(d)   parameters   full Hamiltonian expectation, input-space bias
    SIM   O(p)   parameters   Hamiltonian truncated to p Pauli strings

SIM additionally appears with the quantum branch removed (`SIM-noQ`), which
experiment_circuit_ablation.py showed to be its equivalent hypothesis class.
Including it here checks whether that conclusion survives against HAM and PEFF.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.generators.spectral_pauli_generator import generate_spectral_pauli_strings
from src.models.exact_sim_classifier import ExactSIMClassifier
from src.models.hamiltonian_variants import HAMClassifier, PEFFClassifier
from src.utils.data_loader import (
    load_20newsgroups_projected, load_ecoli_raw, select_topk_chi2)

SEEDS = [0, 1, 2]
EPOCHS = 150
TOP_K = 32


class NoQuantumSIM(ExactSIMClassifier):
    """SIM with <psi|P|psi> replaced by 1 -- the circuit removed entirely."""
    def forward(self, x):
        feats = self.classical_features(x)
        logits = feats @ self.w.T
        return torch.sigmoid(logits.squeeze(-1))


def ecoli(n_qubits, seed):
    X_genes, y = load_ecoli_raw()
    a, b, ytr, yte = train_test_split(X_genes, y, test_size=0.3, random_state=42 + seed)
    Xtr, Xte, _ = select_topk_chi2(a, b, ytr, n_qubits=n_qubits)
    return Xtr, Xte, ytr, yte


def newsgroups(n_qubits, seed):
    X, y = load_20newsgroups_projected(n_qubits)
    return train_test_split(X, y, test_size=0.3, random_state=42 + seed)


DATASETS = [('EColi', 4, ecoli), ('20News', 4, newsgroups)]


def train_eval(model, Xtr, ytr, Xte, yte, epochs=EPOCHS, lr=0.01):
    params = [p for p in model.parameters() if p.requires_grad]
    opt, crit = optim.Adam(params, lr=lr), nn.BCELoss()
    X_t = torch.tensor(Xtr, dtype=torch.float64)
    y_t = torch.tensor(ytr, dtype=torch.float64)
    X_e = torch.tensor(Xte, dtype=torch.float64)

    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        preds = (model(X_e).numpy() > 0.5).astype(int)
    return dict(
        accuracy=accuracy_score(yte, preds),
        balanced=balanced_accuracy_score(yte, preds),
        f1=f1_score(yte, preds, zero_division=0),
        train_loss=loss.item(),
        n_params=sum(p.numel() for p in params),
    )


def run_variant_comparison():
    os.makedirs('results', exist_ok=True)
    rows = []

    for ds_name, n_qubits, loader in DATASETS:
        print(f"\n=== {ds_name} N={n_qubits} ===", flush=True)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = loader(n_qubits, seed)
            basis, _, _ = generate_spectral_pauli_strings(Xtr, ytr, n_qubits)
            basis = basis[:TOP_K]

            builders = {
                'HAM':     lambda: HAMClassifier(n_qubits, n_layers=3),
                'PEFF':    lambda: PEFFClassifier(n_qubits, n_layers=3),
                'SIM':     lambda: ExactSIMClassifier(n_qubits, 3, pauli_strings=basis),
                'SIM-noQ': lambda: NoQuantumSIM(n_qubits, 3, pauli_strings=basis),
            }
            for name, build in builders.items():
                torch.manual_seed(seed)
                np.random.seed(seed)
                m = train_eval(build(), Xtr, ytr, Xte, yte)
                m.update(dataset=ds_name, n_qubits=n_qubits, seed=seed, variant=name)
                rows.append(m)
                print(f"  seed {seed} {name:<8} acc={m['accuracy']:.4f} "
                      f"bal={m['balanced']:.4f} f1={m['f1']:.4f} "
                      f"params={m['n_params']:<6} loss={m['train_loss']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('results/variant_comparison.csv', index=False)

    order = ['HAM', 'PEFF', 'SIM', 'SIM-noQ']
    lines = ["HAM vs PEFF vs SIM (Tiblias et al. Sec. 3.1-3.3)",
             "=" * 78,
             f"Seeds: {len(SEEDS)}  epochs: {EPOCHS}  SIM basis: spectral top-{TOP_K}",
             ""]
    for ds in df['dataset'].unique():
        sub = df[df.dataset == ds]
        maj = None
        lines += [f"{ds}:",
                  f"  {'variant':<9}{'accuracy':>18}{'bal-acc':>11}{'F1':>9}"
                  f"{'params':>9}{'train loss':>13}"]
        for v in order:
            g = sub[sub.variant == v]
            if g.empty:
                continue
            lines.append(f"  {v:<9}{g['accuracy'].mean():>10.4f} +-{g['accuracy'].std():<6.4f}"
                         f"{g['balanced'].mean():>11.4f}{g['f1'].mean():>9.4f}"
                         f"{int(g['n_params'].iloc[0]):>9}{g['train_loss'].mean():>13.4f}")
        lines.append("")

    lines += ["Parameter scaling matches the paper's motivation for the progression:",
              "  HAM  O(N^2) = O(4^n)   PEFF O(d) = O(2^n)   SIM O(p)",
              "",
              "SIM-noQ is SIM with the variational state deleted. If it matches SIM,",
              "the quantum branch is a redundant reparametrization (see",
              "results/circuit_ablation.txt)."]

    report = "\n".join(lines)
    print("\n" + report)
    with open('results/variant_comparison.txt', 'w') as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ['f1', 'accuracy']):
        width, xs = 0.35, np.arange(len(order))
        for i, ds in enumerate(df['dataset'].unique()):
            sub = df[df.dataset == ds]
            means = [sub[sub.variant == v][metric].mean() for v in order]
            errs = [sub[sub.variant == v][metric].std() for v in order]
            ax.bar(xs + i * width, means, width, yerr=errs, capsize=4, label=ds)
        ax.set_xticks(xs + width / 2)
        ax.set_xticklabels(order)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} by Hamiltonian variant')
        ax.legend(); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/variant_comparison.png', dpi=150)
    print("\nSaved results/variant_comparison.{csv,txt,png}")


if __name__ == "__main__":
    run_variant_comparison()
