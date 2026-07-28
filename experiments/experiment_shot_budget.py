"""
The one noise experiment that the w_j absorption argument cannot explain away.

Depolarizing noise rescales each Pauli expectation multiplicatively,
e_j -> lambda_j e_j. Because w_j is a free parameter, w_j' = w_j / lambda_j
recovers the identical function, so the achievable loss is provably unchanged --
any "robustness to depolarizing noise" result is a property of the
parametrization, not evidence about hardware.

Finite sampling is different. Estimating <P_j> from a finite number of shots
gives a *stochastic* error, not a fixed rescale, and no choice of w_j cancels a
random variable. Shots are also the resource that actually costs money on
hardware, which makes this the honest NISQ experiment.

The setup mirrors how SIM would really be deployed. The Pauli expectations do
not depend on the input, so they are measured *once* with a fixed total shot
budget S split evenly over the k selected observables (S/k shots each), and the
classical weights are then fitted against those noisy estimates.

The prediction, stated before running: a random draw of k strings contains only
~k/2 live ones (the rest have x^T P x == 0 identically), so it spends half its
budget estimating numbers that get multiplied by zero. Spectral selection spends
all of S on observables that matter, and should reach a given accuracy at
roughly half the shot cost.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.generators.spectral_pauli_generator import (
    generate_spectral_pauli_strings, is_dead_string)
from src.models.exact_sim_classifier import pauli_observable, real_pauli_stack
from src.utils.data_loader import (
    load_20newsgroups_projected, load_ecoli_raw, select_topk_chi2)
from src.utils.pauli_utils import generate_pauli_strings

K_VALUES = [16, 32, 64]
SHOT_BUDGETS = [1_000, 10_000, 100_000, None]      # None = analytic (infinite shots)
SEEDS = [0, 1, 2]
N_LAYERS = 3


def measure_expectations(pauli_strings, n_qubits, theta, shots, seed):
    """<psi|P_j|psi> for every string, estimated from `shots` samples each."""
    dev = qml.device('default.qubit', wires=n_qubits, shots=shots, seed=seed)

    @qml.qnode(dev)
    def circuit(weights):
        qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
        return [qml.expval(pauli_observable(s)) for s in pauli_strings]

    return np.asarray(circuit(theta), dtype=float)


def fit_and_score(features_tr, y_tr, features_te, y_te):
    """
    With e_j fixed by measurement, Eq. 9 is linear in the scaled features
    (x^T P_j x) * e_j, so the classical stage is plain logistic regression on w.
    """
    sc = StandardScaler().fit(features_tr)
    clf = LogisticRegression(C=10.0, max_iter=2000).fit(sc.transform(features_tr), y_tr)
    p = clf.predict(sc.transform(features_te))
    return (accuracy_score(y_te, p), balanced_accuracy_score(y_te, p),
            f1_score(y_te, p, zero_division=0))


def quadratic_features(X, pauli_strings):
    P = torch.tensor(real_pauli_stack(pauli_strings), dtype=torch.float64)
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
        all_strings = generate_pauli_strings(n_qubits)
        live = [s for s in all_strings if not is_dead_string(s)]
        print(f"\n=== {ds} N={n_qubits} (live {len(live)}/{len(all_strings)}) ===",
              flush=True)

        for seed in SEEDS:
            Xtr, Xte, ytr, yte = loader(n_qubits, seed)
            ranking, _, _ = generate_spectral_pauli_strings(Xtr, ytr, n_qubits)
            rng = np.random.default_rng(500 + seed)
            torch.manual_seed(seed)
            theta = torch.rand(N_LAYERS, n_qubits, 3, dtype=torch.float64)

            for k in K_VALUES:
                bases = {
                    'spectral': ranking[:k],
                    'random_live': [live[i] for i in rng.choice(len(live), k, replace=False)],
                    'random_all': [all_strings[i] for i in
                                   rng.choice(len(all_strings), k, replace=False)],
                }
                for arm, basis in bases.items():
                    n_dead = sum(1 for s in basis if is_dead_string(s))
                    phi_tr = quadratic_features(Xtr, basis)
                    phi_te = quadratic_features(Xte, basis)
                    e_exact = measure_expectations(basis, n_qubits, theta, None, seed)

                    for budget in SHOT_BUDGETS:
                        per_obs = None if budget is None else max(1, budget // k)
                        e_hat = (e_exact if budget is None else
                                 measure_expectations(basis, n_qubits, theta,
                                                      per_obs, seed))
                        est_err = float(np.abs(e_hat - e_exact).mean())

                        acc, bal, f1 = fit_and_score(phi_tr * e_hat, ytr,
                                                     phi_te * e_hat, yte)
                        rows.append(dict(
                            dataset=ds, n_qubits=n_qubits, seed=seed, k=k, arm=arm,
                            budget=(np.inf if budget is None else budget),
                            shots_per_obs=(np.inf if per_obs is None else per_obs),
                            dead_in_basis=n_dead,
                            wasted_shots=(0 if budget is None else n_dead * per_obs),
                            mean_abs_est_error=est_err,
                            accuracy=acc, balanced=bal, f1=f1))
                print(f"  seed {seed} k={k} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('results/shot_budget.csv', index=False)

    lines = ["Shot-budget experiment: finite sampling is not absorbable by w_j",
             "=" * 78,
             f"Seeds: {len(SEEDS)}  |  k: {K_VALUES}  |  budgets: {SHOT_BUDGETS}",
             "",
             "Total budget S is split evenly over the k selected observables.",
             "A dead string (x^T P x == 0 identically) consumes S/k shots and",
             "contributes nothing, so random draws waste roughly half of S.",
             "",
             f"  {'dataset':<9}{'k':>4}{'budget':>10}{'arm':<14}"
             f"{'dead':>6}{'wasted':>9}{'est err':>10}{'F1':>9}"]

    for (ds, k, budget, arm), g in df.groupby(['dataset', 'k', 'budget', 'arm']):
        b = 'analytic' if np.isinf(budget) else f"{int(budget):,}"
        lines.append(f"  {ds:<9}{int(k):>4}{b:>10}{arm:<14}"
                     f"{g['dead_in_basis'].mean():>6.1f}"
                     f"{g['wasted_shots'].mean():>9.0f}"
                     f"{g['mean_abs_est_error'].mean():>10.4f}{g['f1'].mean():>9.4f}")

    lines += ["", "Shot cost to reach the analytic-limit F1 of each arm:"]
    for (ds, k), g in df.groupby(['dataset', 'k']):
        lines.append(f"  {ds} k={k}:")
        for arm in ('spectral', 'random_live', 'random_all'):
            ga = g[g.arm == arm]
            ceiling = ga[np.isinf(ga.budget)]['f1'].mean()
            finite = ga[~np.isinf(ga.budget)].groupby('budget')['f1'].mean()
            reached = [b for b, v in finite.items() if v >= 0.98 * ceiling]
            need = f"{int(min(reached)):,}" if reached else ">100,000"
            lines.append(f"      {arm:<13} ceiling F1 {ceiling:.4f}   "
                         f"shots for 98% of it: {need}")

    report = "\n".join(lines)
    print("\n" + report)
    with open('results/shot_budget.txt', 'w') as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for arm, style in [('spectral', 'o-'), ('random_live', 's--'), ('random_all', '^:')]:
        g = df[(df.arm == arm) & (~np.isinf(df.budget))].groupby('budget')['f1'].mean()
        ax.plot(g.index, g.values, style, label=arm)
        ceil = df[(df.arm == arm) & (np.isinf(df.budget))]['f1'].mean()
        ax.axhline(ceil, ls=':', lw=0.8, alpha=0.4)
    ax.set_xscale('log'); ax.set_xlabel('Total shot budget S'); ax.set_ylabel('F1')
    ax.set_title('Accuracy vs measurement budget (dotted = analytic limit)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    g = df[~np.isinf(df.budget)].groupby(['arm', 'budget'])['wasted_shots'].mean().unstack(0)
    g.plot(kind='bar', ax=ax)
    ax.set_xlabel('Total shot budget S'); ax.set_ylabel('Shots spent on zero observables')
    ax.set_title('Budget wasted on identically-zero observables')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/shot_budget.png', dpi=150)
    print("\nSaved results/shot_budget.{csv,txt,png}")


if __name__ == "__main__":
    run()
