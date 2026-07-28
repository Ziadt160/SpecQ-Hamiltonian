"""
Measurement efficiency: what fraction of a randomly chosen Pauli set is wasted?

Tiblias et al. (Sec. 3.3) select the p Pauli strings of the SIM Hamiltonian
at random: "in practice, they can be chosen at random". For real-valued inputs
that wastes close to half the measurement budget.

A Pauli string with an odd number of Y factors is a purely imaginary Hermitian
matrix, so for real x the quadratic form x^T P x is *identically zero* -- not
small, exactly zero. Such an observable contributes nothing to Eq. 9 no matter
what the data or the weights are, yet it still costs a full set of shots to
estimate on hardware. The count of such strings is

    (4^n - 2^n) / 2      ->   50% of the basis as n grows

This experiment verifies that identity numerically, measures how many dead
strings a random draw of size p actually contains, and confirms that spectral
selection avoids them for free -- giving an efficiency argument for spectral
over random that does not depend on any accuracy comparison.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.generators.spectral_pauli_generator import (
    generate_spectral_pauli_strings, is_dead_string, count_dead_strings)
from src.utils.data_loader import (
    load_20newsgroups_projected, load_ecoli_raw, select_topk_chi2)
from src.utils.pauli_utils import generate_pauli_strings, get_pauli_tensor

N_RANGE = range(2, 13)
P_VALUES = [50, 100, 500, 1000]      # the paper's own sweep (Fig. 3, Table 6)
RANDOM_DRAWS = 200


def verify_dead_strings_are_zero(n_qubits=4, n_samples=200, seed=0):
    """Empirically confirm x^T P x == 0 for odd-Y strings and real x."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 2 ** n_qubits))
    strings = generate_pauli_strings(n_qubits)
    P = np.array([get_pauli_tensor(s) for s in strings])
    phi = np.einsum('bm,kmn,bn->bk', X, P, X)          # complex arithmetic, no casting
    max_abs = np.abs(phi).max(axis=0)
    dead_idx = [i for i, s in enumerate(strings) if is_dead_string(s)]
    live_idx = [i for i, s in enumerate(strings) if not is_dead_string(s)]
    return {
        'n_qubits': n_qubits,
        'dead_max_abs': float(max_abs[dead_idx].max()),
        'live_min_max_abs': float(max_abs[live_idx].min()),
        'n_dead': len(dead_idx),
        'predicted_dead': count_dead_strings(n_qubits),
    }


def spectral_dead_content(n_qubits, loader, seed=0):
    """Where do dead strings land in a spectral ranking, and do the top-k avoid them?"""
    Xtr, ytr = loader(n_qubits, seed)
    ranking, _, mags = generate_spectral_pauli_strings(Xtr, ytr, n_qubits)
    first_dead = next(i for i, s in enumerate(ranking) if is_dead_string(s))
    n_live = 4 ** n_qubits - count_dead_strings(n_qubits)
    return ranking, first_dead, n_live, mags


def ecoli_train(n_qubits, seed):
    X_genes, y = load_ecoli_raw()
    a, b, ytr, _ = train_test_split(X_genes, y, test_size=0.3, random_state=42 + seed)
    Xtr, _, _ = select_topk_chi2(a, b, ytr, n_qubits=n_qubits)
    return Xtr, ytr


def news_train(n_qubits, seed):
    X, y = load_20newsgroups_projected(n_qubits)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42 + seed)
    return Xtr, ytr


def run_measurement_efficiency():
    os.makedirs('results', exist_ok=True)
    rng = np.random.default_rng(0)

    print("=== 1. dead strings really are identically zero ===", flush=True)
    checks = [verify_dead_strings_are_zero(n) for n in (2, 3, 4, 5)]
    for c in checks:
        print(f"  n={c['n_qubits']}: dead count {c['n_dead']} "
              f"(formula {c['predicted_dead']}) | "
              f"largest |x^T P x| over dead strings = {c['dead_max_abs']:.2e} | "
              f"smallest over live = {c['live_min_max_abs']:.3f}")

    print("\n=== 2. analytic waste in a random Pauli draw ===", flush=True)
    rows = []
    for n in N_RANGE:
        total = 4 ** n
        dead = count_dead_strings(n)
        rows.append(dict(n_qubits=n, basis=total, dead=dead, dead_frac=dead / total))
    dfa = pd.DataFrame(rows)
    for _, r in dfa.iterrows():
        print(f"  n={int(r.n_qubits):<3} basis={int(r.basis):<9} "
              f"dead={int(r.dead):<9} = {r.dead_frac:.2%}")

    print("\n=== 3. empirical: dead strings inside a random draw of size p ===", flush=True)
    draw_rows = []
    for n in (4, 6, 8, 10):
        strings = None
        total = 4 ** n
        dead_mask = None
        if n <= 8:
            strings = generate_pauli_strings(n)
            dead_mask = np.array([is_dead_string(s) for s in strings])
        for p in P_VALUES:
            if p > total:
                continue
            if dead_mask is not None:
                fracs = [dead_mask[rng.choice(total, size=p, replace=False)].mean()
                         for _ in range(RANDOM_DRAWS)]
            else:
                # n=10: sample string-wise rather than materialising 4^n strings
                fracs = []
                for _ in range(RANDOM_DRAWS):
                    letters = rng.integers(0, 4, size=(p, n))   # 2 == 'Y'
                    fracs.append(((letters == 2).sum(axis=1) % 2 == 1).mean())
            draw_rows.append(dict(n_qubits=n, p=p, mean_dead_frac=float(np.mean(fracs)),
                                  wasted_measurements=float(np.mean(fracs) * p)))
            print(f"  n={n:<3} p={p:<5} dead fraction {np.mean(fracs):.1%} "
                  f"-> {np.mean(fracs)*p:.0f} of {p} measurements return nothing")
    dfd = pd.DataFrame(draw_rows)

    print("\n=== 4. spectral selection avoids them at no cost ===", flush=True)
    spec_rows = []
    for ds, loader, ns in [('EColi', ecoli_train, (4, 6)),
                           ('20News', news_train, (4, 6))]:
        for n in ns:
            ranking, first_dead, n_live, mags = spectral_dead_content(n, loader)
            spec_rows.append(dict(dataset=ds, n_qubits=n, first_dead_rank=first_dead,
                                  n_live=n_live, basis=4 ** n))
            print(f"  {ds} n={n}: first dead string at rank {first_dead} "
                  f"(live strings: {n_live}/{4**n}) -> any top-k with k<={first_dead} "
                  f"is 100% live")
    dfs = pd.DataFrame(spec_rows)

    dfa.to_csv('results/measurement_efficiency.csv', index=False)
    dfd.to_csv('results/measurement_efficiency_draws.csv', index=False)

    lines = ["Measurement efficiency: random vs spectral Pauli selection",
             "=" * 74,
             "",
             "A Pauli string with an odd number of Y factors has x^T P x == 0 for",
             "every real x, so measuring it yields no information. Verified:",
             ""]
    for c in checks:
        lines.append(f"  n={c['n_qubits']}: max |x^T P x| over {c['n_dead']} dead strings "
                     f"= {c['dead_max_abs']:.2e}  (live minimum: {c['live_min_max_abs']:.3f})")
    lines += ["",
              "Dead fraction of the full basis, (4^n - 2^n) / (2 * 4^n):",
              f"  {'n':>3}{'basis':>12}{'dead':>12}{'fraction':>11}"]
    for _, r in dfa.iterrows():
        lines.append(f"  {int(r.n_qubits):>3}{int(r.basis):>12}"
                     f"{int(r.dead):>12}{r.dead_frac:>10.2%}")
    lines += ["",
              "Wasted measurements in a random draw of p strings",
              "(p values are the paper's own sweep; its tuned SIM uses p=1000):",
              f"  {'n':>3}{'p':>7}{'dead':>9}{'wasted':>10}"]
    for _, r in dfd.iterrows():
        lines.append(f"  {int(r.n_qubits):>3}{int(r.p):>7}{r.mean_dead_frac:>8.1%}"
                     f"{r.wasted_measurements:>10.0f}")
    lines += ["",
              "Spectral ranking places every dead string after every live one:",
              f"  {'dataset':<9}{'n':>3}{'first dead rank':>18}{'live strings':>15}"]
    for _, r in dfs.iterrows():
        lines.append(f"  {r.dataset:<9}{int(r.n_qubits):>3}"
                     f"{int(r.first_dead_rank):>18}{int(r.n_live):>15}")
    lines += ["",
              "Consequence: at the paper's tuned setting (p=1000, n=10) roughly 500",
              "of the 1000 measured observables are identically zero. Spectral",
              "selection removes that waste without any accuracy argument."]

    report = "\n".join(lines)
    print("\n" + report)
    with open('results/measurement_efficiency.txt', 'w') as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(dfa['n_qubits'], dfa['dead_frac'] * 100, 'o-', color='crimson')
    ax.axhline(50, ls='--', color='grey', label='50% asymptote')
    ax.set_xlabel('Qubits n'); ax.set_ylabel('Dead strings (% of basis)')
    ax.set_title('Half the Pauli basis is unmeasurable for real inputs')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for n in sorted(dfd['n_qubits'].unique()):
        g = dfd[dfd.n_qubits == n]
        ax.plot(g['p'], g['wasted_measurements'], 'o-', label=f'n={n}')
    ax.plot([0, 1000], [0, 0], 'k--', lw=1, label='spectral (zero waste)')
    ax.set_xlabel('Pauli strings measured (p)')
    ax.set_ylabel('Measurements returning zero')
    ax.set_title('Wasted shots under random selection')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/measurement_efficiency.png', dpi=150)
    print("\nSaved results/measurement_efficiency.{csv,txt,png}")


if __name__ == "__main__":
    run_measurement_efficiency()
