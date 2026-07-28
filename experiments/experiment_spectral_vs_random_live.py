"""
Does the spectral *ranking* carry information, or is it just a dead-string filter?

Every spectral-vs-random comparison in this repository compares against strings
drawn from the *full* Pauli basis. But roughly half that basis is identically
zero for real inputs (see experiment_measurement_efficiency.py), so a random
draw of size k contains only ~k/2 usable observables while a spectral top-k
contains k. That comparison therefore conflates two different claims:

  (a) "don't measure observables that are identically zero"   -- trivial, provable
  (b) "rank the remaining observables by spectral energy"     -- the actual method

This experiment separates them with three arms at equal k:

  spectral      top-k by |c_P|                     (the method)
  random_live   k drawn from live strings only     (controls for (a))
  random_all    k drawn from the whole basis       (what the repo compared against)

spectral vs random_all measures (a) + (b) together.
spectral vs random_live measures (b) alone -- the honest test of the ranking.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.generators.spectral_pauli_generator import (
    generate_spectral_pauli_strings, is_dead_string)
from src.models.sim_classifier import SIMClassifier
from src.utils.data_loader import (
    load_20newsgroups_projected, load_ecoli_raw, select_topk_chi2)
from src.utils.pauli_utils import generate_pauli_strings

K_VALUES = [8, 16, 32, 64, 128]
SEEDS = [0, 1, 2, 3, 4]
DRAWS_PER_SEED = 5          # average the random arms over several draws


def ecoli(n_qubits, seed):
    X_genes, y = load_ecoli_raw()
    a, b, ytr, yte = train_test_split(X_genes, y, test_size=0.3, random_state=42 + seed)
    Xtr, Xte, _ = select_topk_chi2(a, b, ytr, n_qubits=n_qubits)
    return Xtr, Xte, ytr, yte


def newsgroups(n_qubits, seed):
    X, y = load_20newsgroups_projected(n_qubits)
    return train_test_split(X, y, test_size=0.3, random_state=42 + seed)


DATASETS = [('EColi', 4, ecoli), ('EColi', 6, ecoli),
            ('20News', 4, newsgroups), ('20News', 6, newsgroups)]


def score(basis, Xtr, ytr, Xte, yte):
    sim = SIMClassifier(pauli_strings=list(basis), C=10.0, random_state=42)
    sim.fit(Xtr, ytr)
    p = sim.predict(Xte)
    return (accuracy_score(yte, p), balanced_accuracy_score(yte, p),
            f1_score(yte, p, zero_division=0))


def run():
    os.makedirs('results', exist_ok=True)
    rows = []

    for ds, n_qubits, loader in DATASETS:
        all_strings = generate_pauli_strings(n_qubits)
        live = [s for s in all_strings if not is_dead_string(s)]
        print(f"\n=== {ds} N={n_qubits} "
              f"(basis {len(all_strings)}, live {len(live)}) ===", flush=True)

        for seed in SEEDS:
            Xtr, Xte, ytr, yte = loader(n_qubits, seed)
            ranking, _, _ = generate_spectral_pauli_strings(Xtr, ytr, n_qubits)
            rng = np.random.default_rng(1000 + seed)

            for k in K_VALUES:
                if k > len(live):
                    continue

                acc, bal, f1 = score(ranking[:k], Xtr, ytr, Xte, yte)
                rows.append(dict(dataset=ds, n_qubits=n_qubits, seed=seed, k=k,
                                 arm='spectral', accuracy=acc, balanced=bal, f1=f1))

                for arm, pool in (('random_live', live), ('random_all', all_strings)):
                    accs, bals, f1s = [], [], []
                    for _ in range(DRAWS_PER_SEED):
                        pick = rng.choice(len(pool), size=k, replace=False)
                        a_, b_, f_ = score([pool[i] for i in pick], Xtr, ytr, Xte, yte)
                        accs.append(a_); bals.append(b_); f1s.append(f_)
                    rows.append(dict(dataset=ds, n_qubits=n_qubits, seed=seed, k=k,
                                     arm=arm, accuracy=np.mean(accs),
                                     balanced=np.mean(bals), f1=np.mean(f1s)))
            print(f"  seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('results/spectral_vs_random_live.csv', index=False)

    piv = df.pivot_table(index=['dataset', 'n_qubits', 'k'],
                         columns='arm', values='f1').reset_index()

    lines = ["Is the spectral ranking informative, or just a dead-string filter?",
             "=" * 78,
             f"Seeds: {len(SEEDS)}  random draws per seed: {DRAWS_PER_SEED}  metric: F1",
             "",
             "  random_all  = k drawn from the full basis (~half of them dead)",
             "  random_live = k drawn from live strings only",
             "  spectral    = top-k by spectral energy",
             "",
             f"  {'dataset':<9}{'n':>3}{'k':>6}{'spectral':>11}{'rnd_live':>11}"
             f"{'rnd_all':>10}{'vs live':>10}{'vs all':>9}"]

    for _, r in piv.iterrows():
        lines.append(f"  {r['dataset']:<9}{int(r['n_qubits']):>3}{int(r['k']):>6}"
                     f"{r['spectral']:>11.4f}{r['random_live']:>11.4f}"
                     f"{r['random_all']:>10.4f}"
                     f"{r['spectral'] - r['random_live']:>+10.4f}"
                     f"{r['spectral'] - r['random_all']:>+9.4f}")

    gain_live = (piv['spectral'] - piv['random_live'])
    gain_all = (piv['spectral'] - piv['random_all'])
    dead_only = (piv['random_live'] - piv['random_all'])

    lines += ["",
              "Decomposition of the advantage spectral holds over a naive random draw:",
              f"  spectral - random_all   (total)            {gain_all.mean():+.4f}",
              f"    of which:",
              f"      random_live - random_all (dead filter)  {dead_only.mean():+.4f}"
              f"   ({100 * dead_only.mean() / gain_all.mean():.0f}% of the total)",
              f"      spectral - random_live   (ranking)      {gain_live.mean():+.4f}"
              f"   ({100 * gain_live.mean() / gain_all.mean():.0f}% of the total)",
              "",
              f"Configurations where the ranking beats random-among-live: "
              f"{int((gain_live > 0).sum())}/{len(gain_live)}"]

    report = "\n".join(lines)
    print("\n" + report)
    with open('results/spectral_vs_random_live.txt', 'w') as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for arm, style in [('spectral', 'o-'), ('random_live', 's--'), ('random_all', '^:')]:
        g = df[df.arm == arm].groupby('k')['f1'].mean()
        ax.plot(g.index, g.values, style, label=arm)
    ax.set_xscale('log', base=2); ax.set_xlabel('Basis size k'); ax.set_ylabel('F1')
    ax.set_title('Spectral vs random, with and without dead strings')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ks = sorted(piv['k'].unique())
    dead_by_k = [(piv[piv.k == k]['random_live'] - piv[piv.k == k]['random_all']).mean()
                 for k in ks]
    rank_by_k = [(piv[piv.k == k]['spectral'] - piv[piv.k == k]['random_live']).mean()
                 for k in ks]
    ax.bar(range(len(ks)), dead_by_k, label='dead-string filter', color='#dd8452')
    ax.bar(range(len(ks)), rank_by_k, bottom=dead_by_k, label='spectral ranking',
           color='#4c72b0')
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks)
    ax.set_xlabel('Basis size k'); ax.set_ylabel('F1 gain over random_all')
    ax.set_title('Where the advantage comes from')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/spectral_vs_random_live.png', dpi=150)
    print("\nSaved results/spectral_vs_random_live.{csv,txt,png}")


if __name__ == "__main__":
    run()
