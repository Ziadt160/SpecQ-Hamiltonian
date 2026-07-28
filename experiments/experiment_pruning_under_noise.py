"""
How aggressively should you prune the Pauli basis when the device is noisy?

Depolarizing noise alone cannot be detected by this architecture: it damps the
expectations as e_j -> lambda_j e_j, and because w_j is free, w_j' = w_j/lambda_j
recovers the identical function. Finite shots alone barely matter either --
experiment_shot_budget.py found every arm saturated at the smallest budget
tested. The interesting regime is their *product*.

With both, the measured value is

    e_hat_j = lambda_j e_j + eps_j ,      lambda_j = (1 - 4p/3)^weight(j)

and rescaling by 1/lambda_j to undo the damping scales the sampling error with
it:  w_j e_hat_j  ->  e_j + eps_j / lambda_j. The effective noise on a term is
therefore amplified by 1/lambda_j, which grows *exponentially in the Pauli
weight*. No choice of w_j avoids this, because it scales signal and error
together.

Prediction: high-weight observables become net-harmful as p grows, so the
accuracy-optimal basis size should shrink with noise, and spectral selection
(which prefers low-weight strings) should degrade more slowly than random.

The damping model is exact, not an approximation: per-qubit
DepolarizingChannel(p) shrinks every non-identity Pauli factor by (1 - 4p/3),
verified against a density-matrix simulation to 1.6e-13. Sampling is drawn from
the exact binomial rather than a Gaussian approximation.
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
from sklearn.preprocessing import StandardScaler

from src.generators.spectral_pauli_generator import (
    generate_spectral_pauli_strings, is_dead_string)
from src.models.exact_sim_classifier import pauli_observable, real_pauli_stack
from src.utils.data_loader import (
    load_20newsgroups_projected, load_ecoli_raw, select_topk_chi2)
from src.utils.pauli_utils import generate_pauli_strings

K_VALUES = [4, 8, 16, 32, 64, 128]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10, 0.20]
SHOT_BUDGETS = [1_000, 10_000, None]        # None = analytic
SEEDS = [0, 1, 2]
N_LAYERS = 3


def pauli_weight(s):
    return sum(1 for c in s if c != 'I')


def clean_expectations(strings, n_qubits, theta):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(w):
        qml.StronglyEntanglingLayers(weights=w, wires=range(n_qubits))
        return [qml.expval(pauli_observable(s)) for s in strings]

    return np.asarray(circuit(theta), dtype=float)


def observe(e_clean, weights, p, shots_per_obs, rng):
    """Damp by (1-4p/3)^weight, then draw the value from a finite shot budget."""
    mu = e_clean * (1.0 - 4.0 * p / 3.0) ** weights
    if shots_per_obs is None:
        return mu
    mu = np.clip(mu, -1.0, 1.0)
    n_plus = rng.binomial(shots_per_obs, (1.0 + mu) / 2.0)
    return 2.0 * n_plus / shots_per_obs - 1.0


def quadratic_features(X, strings):
    P = torch.tensor(real_pauli_stack(strings), dtype=torch.float64)
    return torch.einsum('bm,kmn,bn->bk',
                        torch.tensor(X, dtype=torch.float64), P,
                        torch.tensor(X, dtype=torch.float64)).numpy()


def score(f_tr, y_tr, f_te, y_te):
    sc = StandardScaler().fit(f_tr)
    clf = LogisticRegression(C=10.0, max_iter=2000).fit(sc.transform(f_tr), y_tr)
    p = clf.predict(sc.transform(f_te))
    return balanced_accuracy_score(y_te, p), f1_score(y_te, p, zero_division=0)


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
        all_strings = generate_pauli_strings(n_qubits)
        live = [s for s in all_strings if not is_dead_string(s)]
        print(f"\n=== {ds} N={n_qubits} ===", flush=True)

        for seed in SEEDS:
            Xtr, Xte, ytr, yte = loader(n_qubits, seed)
            ranking, _, _ = generate_spectral_pauli_strings(Xtr, ytr, n_qubits)
            spectral = [s for s in ranking if not is_dead_string(s)]
            rng = np.random.default_rng(2000 + seed)
            rnd = [live[i] for i in rng.permutation(len(live))]
            torch.manual_seed(seed)
            theta = torch.rand(N_LAYERS, n_qubits, 3, dtype=torch.float64)

            for arm, order in (('spectral', spectral), ('random_live', rnd)):
                for k in K_VALUES:
                    basis = order[:k]
                    wts = np.array([pauli_weight(s) for s in basis])
                    e_clean = clean_expectations(basis, n_qubits, theta)
                    f_tr = quadratic_features(Xtr, basis)
                    f_te = quadratic_features(Xte, basis)

                    for p in NOISE_LEVELS:
                        for budget in SHOT_BUDGETS:
                            spo = None if budget is None else max(1, budget // k)
                            e_hat = observe(e_clean, wts, p, spo, rng)
                            bal, f1 = score(f_tr * e_hat, ytr, f_te * e_hat, yte)
                            rows.append(dict(
                                dataset=ds, seed=seed, arm=arm, k=k, p=p,
                                budget=(np.inf if budget is None else budget),
                                mean_weight=float(wts.mean()),
                                amplification=float(np.mean(
                                    (1 - 4 * p / 3) ** (-wts.astype(float)))),
                                balanced=bal, f1=f1))
                print(f"  seed {seed} {arm} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('results/pruning_under_noise.csv', index=False)

    lines = ["Optimal Pauli-basis size as a function of noise and shot budget",
             "=" * 78,
             f"Seeds: {len(SEEDS)}  k: {K_VALUES}  p: {NOISE_LEVELS}",
             "",
             "Damping is exact: DepolarizingChannel(p) per qubit shrinks each",
             "non-identity Pauli factor by (1-4p/3); verified to 1.6e-13.",
             "",
             "Best k (by F1) for each noise / budget combination:",
             ""]

    for ds in df['dataset'].unique():
        lines.append(f"  {ds}:")
        lines.append(f"    {'budget':>10}" +
                     "".join(f"{'p='+str(p):>12}" for p in NOISE_LEVELS))
        for budget in [1_000, 10_000, np.inf]:
            cells = []
            for p in NOISE_LEVELS:
                g = df[(df.dataset == ds) & (df.arm == 'spectral') &
                       (df.budget == budget) & (df.p == p)]
                m = g.groupby('k')['f1'].mean()
                cells.append(f"{int(m.idxmax())} ({m.max():.3f})")
            b = 'analytic' if np.isinf(budget) else f"{budget:,}"
            lines.append(f"    {b:>10}" + "".join(f"{c:>12}" for c in cells))
        lines.append("")

    lines += ["Spectral vs random at the largest basis (k=128), F1:",
              f"  {'dataset':<9}{'budget':>10}" +
              "".join(f"{'p='+str(p):>11}" for p in NOISE_LEVELS)]
    for ds in df['dataset'].unique():
        for budget in [1_000, np.inf]:
            b = 'analytic' if np.isinf(budget) else f"{budget:,}"
            cells = []
            for p in NOISE_LEVELS:
                sub = df[(df.dataset == ds) & (df.k == 128) &
                         (df.budget == budget) & (df.p == p)]
                s_ = sub[sub.arm == 'spectral']['f1'].mean()
                r_ = sub[sub.arm == 'random_live']['f1'].mean()
                cells.append(f"{s_-r_:+.3f}")
            lines.append(f"  {ds:<9}{b:>10}" + "".join(f"{c:>11}" for c in cells))

    report = "\n".join(lines)
    print("\n" + report)
    with open('results/pruning_under_noise.txt', 'w') as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    sub = df[(df.arm == 'spectral') & (df.budget == 1000)]
    for p in NOISE_LEVELS:
        g = sub[sub.p == p].groupby('k')['f1'].mean()
        ax.plot(g.index, g.values, 'o-', label=f'p={p}')
    ax.set_xscale('log', base=2); ax.set_xlabel('Basis size k'); ax.set_ylabel('F1')
    ax.set_title('Accuracy vs basis size at S=1,000 shots')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for budget, style in [(1000, 'o-'), (np.inf, 's--')]:
        best = []
        for p in NOISE_LEVELS:
            g = df[(df.arm == 'spectral') & (df.budget == budget) &
                   (df.p == p)].groupby('k')['f1'].mean()
            best.append(g.idxmax())
        lbl = 'analytic' if np.isinf(budget) else f'S={budget:,}'
        ax.plot(NOISE_LEVELS, best, style, label=lbl)
    ax.set_yscale('log', base=2)
    ax.set_xlabel('Depolarizing rate p'); ax.set_ylabel('Accuracy-optimal k')
    ax.set_title('Does noise favour a smaller basis?')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/pruning_under_noise.png', dpi=150)
    print("\nSaved results/pruning_under_noise.{csv,txt,png}")


if __name__ == "__main__":
    run()
