"""
Which Delta does Spectral Pauli Pruning actually want?

The paper (Eq. 4) specifies

    Delta = Cov(X | y=1) - Cov(X | y=0)

but the implementation computes class-conditional *second moments*

    Delta = E[xx^T | y=1] - E[xx^T | y=0]

with no mean subtraction. These agree only when both class-conditional means
vanish, which they do not for any dataset here. The two definitions disagree on
roughly a third of the selected top-32 basis, so this is not a cosmetic
difference -- one of the paper and the code has to change, and this experiment
decides which way.

Selection quality is scored with SIMClassifier (regularized logistic regression
on the quadratic Pauli features). That is the right probe: experiment_circuit_
ablation.py shows the variational state contributes no expressivity, so the
classical model measures basis quality directly, without the optimizer noise
and 100x cost of a QNode training run.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.generators.spectral_pauli_generator import (
    generate_spectral_pauli_strings, compute_delta, VALID_MOMENTS)
from src.models.sim_classifier import SIMClassifier
from src.utils.data_loader import (
    load_20newsgroups_projected, load_ecoli_raw, select_topk_chi2)

K_VALUES = [8, 16, 32, 50, 64, 128]
SEEDS = [0, 1, 2, 3, 4]


def ecoli_split(n_qubits, seed):
    X_genes, y = load_ecoli_raw()
    a, b, ytr, yte = train_test_split(X_genes, y, test_size=0.3, random_state=42 + seed)
    Xtr, Xte, _ = select_topk_chi2(a, b, ytr, n_qubits=n_qubits)
    return Xtr, Xte, ytr, yte


def newsgroups_split(n_qubits, seed):
    X, y = load_20newsgroups_projected(n_qubits)
    return train_test_split(X, y, test_size=0.3, random_state=42 + seed)


DATASETS = [
    ('EColi', 4, ecoli_split),
    ('EColi', 6, ecoli_split),
    ('20News', 4, newsgroups_split),
    ('20News', 6, newsgroups_split),
]


def score_basis(basis, Xtr, ytr, Xte, yte):
    sim = SIMClassifier(pauli_strings=basis, C=10.0, random_state=42)
    sim.fit(Xtr, ytr)
    p = sim.predict(Xte)
    return (accuracy_score(yte, p),
            balanced_accuracy_score(yte, p),
            f1_score(yte, p, average='binary', zero_division=0))


def run_delta_comparison():
    os.makedirs('results', exist_ok=True)
    rows, overlap_rows = [], []

    for ds_name, n_qubits, loader in DATASETS:
        print(f"\n=== {ds_name} N={n_qubits} ===", flush=True)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = loader(n_qubits, seed)

            rankings = {}
            for moment in VALID_MOMENTS:
                strings, _, _ = generate_spectral_pauli_strings(
                    Xtr, ytr, n_qubits, moment=moment)
                rankings[moment] = strings

            for k in K_VALUES:
                a = set(rankings['second_moment'][:k])
                b = set(rankings['covariance'][:k])
                overlap_rows.append(dict(dataset=ds_name, n_qubits=n_qubits,
                                         seed=seed, k=k,
                                         overlap=len(a & b) / k))
                for moment in VALID_MOMENTS:
                    acc, bal, f1 = score_basis(rankings[moment][:k], Xtr, ytr, Xte, yte)
                    rows.append(dict(dataset=ds_name, n_qubits=n_qubits, seed=seed,
                                     k=k, moment=moment,
                                     accuracy=acc, balanced=bal, f1=f1))
            print(f"  seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    ov = pd.DataFrame(overlap_rows)
    df.to_csv('results/delta_definition.csv', index=False)

    lines = ["Delta definition: covariance (paper Eq. 4) vs second moment (code)",
             "=" * 78,
             f"Seeds: {len(SEEDS)}  |  basis sizes: {K_VALUES}",
             "",
             f"{'dataset':<9}{'n':>3}{'k':>6}{'overlap':>10}"
             f"{'2nd-mom F1':>13}{'cov F1':>10}{'delta':>9}"]

    for (ds, nq, k), g in df.groupby(['dataset', 'n_qubits', 'k'], sort=False):
        s = g[g.moment == 'second_moment']['f1'].mean()
        c = g[g.moment == 'covariance']['f1'].mean()
        o = ov[(ov.dataset == ds) & (ov.n_qubits == nq) & (ov.k == k)]['overlap'].mean()
        lines.append(f"{ds:<9}{nq:>3}{k:>6}{o:>9.1%}{s:>13.4f}{c:>10.4f}{c - s:>+9.4f}")

    lines += ["", "Aggregate over every dataset/size/seed:"]
    for metric in ('accuracy', 'balanced', 'f1'):
        s = df[df.moment == 'second_moment'][metric].mean()
        c = df[df.moment == 'covariance'][metric].mean()
        lines.append(f"  mean {metric:<10} second_moment={s:.4f}  covariance={c:.4f}  "
                     f"(cov - 2nd = {c - s:+.4f})")

    wins = df.pivot_table(index=['dataset', 'n_qubits', 'seed', 'k'],
                          columns='moment', values='f1')
    n_cov = int((wins['covariance'] > wins['second_moment']).sum())
    n_sec = int((wins['second_moment'] > wins['covariance']).sum())
    lines += ["",
              f"Per-configuration F1 wins: covariance {n_cov}, second_moment {n_sec}, "
              f"ties {len(wins) - n_cov - n_sec}"]

    report = "\n".join(lines)
    print("\n" + report)
    with open('results/delta_definition.txt', 'w') as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for moment, style in [('second_moment', 'o-'), ('covariance', 's--')]:
        g = df[df.moment == moment].groupby('k')['f1'].mean()
        ax.plot(g.index, g.values, style, label=moment)
    ax.set_xscale('log', base=2); ax.set_xlabel('Basis size k'); ax.set_ylabel('F1')
    ax.set_title('Basis quality by Delta definition'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    g = ov.groupby('k')['overlap'].mean()
    ax.plot(g.index, g.values * 100, 'o-', color='crimson')
    ax.set_xscale('log', base=2); ax.set_xlabel('Basis size k')
    ax.set_ylabel('Top-k overlap (%)')
    ax.set_title('How much the two definitions disagree'); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/delta_definition.png', dpi=150)
    print("\nSaved results/delta_definition.{csv,txt,png}")


if __name__ == "__main__":
    run_delta_comparison()
