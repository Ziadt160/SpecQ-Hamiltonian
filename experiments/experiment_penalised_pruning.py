"""
Noise-aware basis selection when the weights are not free.

The reason noise cannot be detected in the usual setup is that w_j is
unconstrained: noise damps e_j -> lambda_j e_j, and training simply grows
w_j -> w_j/lambda_j to cancel it. Worse, any pipeline that standardises the
Pauli features removes e_j *exactly*, because e_j is a constant multiplier on
column j and StandardScaler divides each column by its own standard deviation.
Verified: damping every expectation by 1e-6, or flipping every sign, leaves the
fitted decision function identical to 1e-12.

So this experiment removes both escapes:

  * the classical features phi_j are standardised ONCE, on the noiseless data,
    and that fixed scaling is reused at every noise level -- so the only thing
    changing with p is the e_j damping, not the feature scale;
  * no further standardisation is applied after multiplying by e_hat_j, so the
    damping is visible to the model;
  * an L2 penalty acts directly on w. Undoing a damping factor lambda_j now
    costs penalty proportional to 1/lambda_j^2, which grows exponentially in the
    Pauli weight.

Under those conditions a heavily damped, high-weight observable is genuinely
expensive to keep, and the accuracy-optimal basis should shrink as p grows.

This tests a claim about *how to choose the basis under a resource constraint*.
It is not a claim that the model resists noise -- that one is dead either way.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.generators.spectral_pauli_generator import (
    generate_spectral_pauli_strings, is_dead_string)
from src.models.exact_sim_classifier import pauli_observable, real_pauli_stack
from src.utils.data_loader import (
    load_20newsgroups_projected, load_ecoli_raw, select_topk_chi2)

K_VALUES = [4, 8, 16, 32, 64, 128]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
C_VALUES = [0.01, 0.1, 1.0]          # smaller C = stronger penalty on w
SHOTS = 10_000
SEEDS = [0, 1, 2]
N_LAYERS = 3


def clean_expectations(strings, n_qubits, theta):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(w):
        qml.StronglyEntanglingLayers(weights=w, wires=range(n_qubits))
        return [qml.expval(pauli_observable(s)) for s in strings]

    return np.asarray(circuit(theta), dtype=float)


def observe(e_clean, weights, p, shots, rng):
    """Exact depolarizing damping (1-4p/3)^weight, then a finite shot draw."""
    mu = np.clip(e_clean * (1.0 - 4.0 * p / 3.0) ** weights, -1.0, 1.0)
    if shots is None:
        return mu
    return 2.0 * rng.binomial(shots, (1.0 + mu) / 2.0) / shots - 1.0


def quadratic_features(X, strings):
    P = torch.tensor(real_pauli_stack(strings), dtype=torch.float64)
    Xt = torch.tensor(X, dtype=torch.float64)
    return torch.einsum('bm,kmn,bn->bk', Xt, P, Xt).numpy()


def ecoli(n_qubits, seed):
    Xg, y = load_ecoli_raw()
    a, b, ytr, yte = train_test_split(Xg, y, test_size=0.3, random_state=42 + seed)
    Xtr, Xte, _ = select_topk_chi2(a, b, ytr, n_qubits=n_qubits)
    return Xtr, Xte, ytr, yte


def newsgroups(n_qubits, seed):
    X, y = load_20newsgroups_projected(n_qubits)
    return train_test_split(X, y, test_size=0.3, random_state=42 + seed)


DATASETS = [('EColi', 4, ecoli), ('20News', 4, newsgroups)]


def run():
    os.makedirs('results', exist_ok=True)
    rows = []

    for ds, n_qubits, loader in DATASETS:
        print(f"\n=== {ds} N={n_qubits} ===", flush=True)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = loader(n_qubits, seed)
            ranking, _, _ = generate_spectral_pauli_strings(Xtr, ytr, n_qubits)
            basis_order = [s for s in ranking if not is_dead_string(s)]
            rng = np.random.default_rng(3000 + seed)
            torch.manual_seed(seed)
            theta = torch.rand(N_LAYERS, n_qubits, 3, dtype=torch.float64)

            for k in K_VALUES:
                basis = basis_order[:k]
                wts = np.array([sum(1 for c in s if c != 'I') for s in basis])
                e_clean = clean_expectations(basis, n_qubits, theta)

                # Fixed feature scaling, computed once on the noiseless features
                # so that changing p changes only the damping, never the scale.
                f_tr = quadratic_features(Xtr, basis)
                f_te = quadratic_features(Xte, basis)
                scale = f_tr.std(axis=0)
                scale[scale == 0] = 1.0
                mu_tr = f_tr.mean(axis=0)
                f_tr = (f_tr - mu_tr) / scale
                f_te = (f_te - mu_tr) / scale

                for p in NOISE_LEVELS:
                    e_hat = observe(e_clean, wts, p, SHOTS, rng)
                    Ztr, Zte = f_tr * e_hat, f_te * e_hat
                    for C in C_VALUES:
                        # No further standardisation: the L2 penalty now sees
                        # the damping, so cancelling it costs 1/lambda^2.
                        clf = LogisticRegression(C=C, max_iter=5000).fit(Ztr, ytr)
                        pred = clf.predict(Zte)
                        rows.append(dict(
                            dataset=ds, seed=seed, k=k, p=p, C=C,
                            mean_weight=float(wts.mean()),
                            coef_norm=float(np.linalg.norm(clf.coef_)),
                            balanced=balanced_accuracy_score(yte, pred),
                            f1=f1_score(yte, pred, zero_division=0)))
            print(f"  seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('results/penalised_pruning.csv', index=False)

    lines = ["Noise-aware basis size when w carries an L2 penalty",
             "=" * 78,
             f"Seeds: {len(SEEDS)}  k: {K_VALUES}  p: {NOISE_LEVELS}  shots: {SHOTS:,}",
             "",
             "Classical features are standardised once on the noiseless data and that",
             "scaling is reused, so p changes only the damping. No scaling is applied",
             "after multiplying by e_hat, so the L2 penalty on w sees it.",
             "",
             "Accuracy-optimal k (F1) per noise level:"]

    for ds in df['dataset'].unique():
        lines.append(f"\n  {ds}:")
        lines.append(f"    {'C':>7}" + "".join(f"{'p='+str(p):>13}" for p in NOISE_LEVELS))
        for C in C_VALUES:
            cells = []
            for p in NOISE_LEVELS:
                g = df[(df.dataset == ds) & (df.C == C) & (df.p == p)]
                m = g.groupby('k')['f1'].mean()
                cells.append(f"{int(m.idxmax())} ({m.max():.3f})")
            lines.append(f"    {C:>7}" + "".join(f"{c:>13}" for c in cells))

    lines += ["", "Fitted ||w|| at k=128 (compensation effort vs noise):",
              f"  {'dataset':<9}{'C':>7}" + "".join(f"{'p='+str(p):>11}" for p in NOISE_LEVELS)]
    for ds in df['dataset'].unique():
        for C in C_VALUES:
            cells = [f"{df[(df.dataset==ds)&(df.C==C)&(df.p==p)&(df.k==128)]['coef_norm'].mean():.2f}"
                     for p in NOISE_LEVELS]
            lines.append(f"  {ds:<9}{C:>7}" + "".join(f"{c:>11}" for c in cells))

    report = "\n".join(lines)
    print("\n" + report)
    with open('results/penalised_pruning.txt', 'w') as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    sub = df[(df.dataset == 'EColi') & (df.C == 0.01)]
    for p in NOISE_LEVELS:
        g = sub[sub.p == p].groupby('k')['f1'].mean()
        ax.plot(g.index, g.values, 'o-', label=f'p={p}')
    ax.set_xscale('log', base=2); ax.set_xlabel('Basis size k'); ax.set_ylabel('F1')
    ax.set_title('E. Coli, strong penalty (C=0.01)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for ds in df['dataset'].unique():
        for C in C_VALUES:
            best = [df[(df.dataset==ds)&(df.C==C)&(df.p==p)].groupby('k')['f1'].mean().idxmax()
                    for p in NOISE_LEVELS]
            ax.plot(NOISE_LEVELS, best, 'o-', label=f'{ds} C={C}')
    ax.set_yscale('log', base=2)
    ax.set_xlabel('Depolarizing rate p'); ax.set_ylabel('Accuracy-optimal k')
    ax.set_title('Does a penalised model prune harder under noise?')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/penalised_pruning.png', dpi=150)
    print("\nSaved results/penalised_pruning.{csv,txt,png}")


if __name__ == "__main__":
    run()
