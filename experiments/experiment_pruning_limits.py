"""
How far can the Pauli basis be pruned, and do extra strings cause overfitting?

Two questions on one axis:

  (1) Pruning limit. What is the smallest k that still reaches a given fraction
      of full-basis accuracy? Reported as the minimum k retaining 90 / 95 / 99%
      of the full-basis test F1.

  (2) Overfitting. Each retained string adds exactly one parameter w_j, so the
      model capacity is k. At n=6 the full live basis is 2080 strings against
      ~1350 training samples, so the sweep crosses k = N. If extra strings
      overfit, the train-test gap should open there.

The classical path (SIMClassifier: regularized logistic regression on the
quadratic Pauli features) is used throughout. That is not an approximation --
the variational state provably contributes nothing once the features are
standardised, since e_j is constant across samples and is divided out exactly.
It also runs fast enough to sweep k finely to the full basis, which the QNode
path cannot.

Both spectral and random-from-live orderings are swept, so the overfitting
question can be asked separately of "more strings" and "worse strings".
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.generators.spectral_pauli_generator import (
    generate_spectral_pauli_strings, is_dead_string)
from src.models.exact_sim_classifier import real_pauli_stack
from src.utils.data_loader import (
    load_20newsgroups_projected, load_ecoli_raw, select_topk_chi2)
from src.utils.pauli_utils import generate_pauli_strings

FEATURE_CHUNK = 256          # Pauli strings materialised at a time

SEEDS = [0, 1, 2, 3, 4]
RETENTION_TARGETS = [0.90, 0.95, 0.99]


def k_grid(n_live):
    """Fine at the small end, log-spaced after, always including the full basis."""
    small = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    grid = [k for k in small if k <= n_live]
    k = 192
    while k < n_live:
        grid.append(k)
        k = int(k * 1.5)
    grid.append(n_live)
    return sorted(set(grid))


def ecoli(n_qubits, seed):
    Xg, y = load_ecoli_raw()
    a, b, ytr, yte = train_test_split(Xg, y, test_size=0.3, random_state=42 + seed)
    Xtr, Xte, _ = select_topk_chi2(a, b, ytr, n_qubits=n_qubits)
    return Xtr, Xte, ytr, yte


def newsgroups(n_qubits, seed):
    X, y = load_20newsgroups_projected(n_qubits)
    return train_test_split(X, y, test_size=0.3, random_state=42 + seed)


DATASETS = [('EColi', 4, ecoli), ('EColi', 6, ecoli),
            ('20News', 4, newsgroups), ('20News', 6, newsgroups)]


def quadratic_features(X, strings):
    """
    phi_j(x) = x^T P_j x for every string, in the given order.

    Materialises the Pauli stack in chunks: the full n=6 basis is 2080 matrices
    of 64x64, and building them all at once (as SIMClassifier does, in complex128)
    exhausts memory when the sweep instantiates a fresh model per basis size.
    Computing once and slicing columns is also far faster than recomputing.
    """
    Xt = torch.tensor(X, dtype=torch.float64)
    out = []
    for i in range(0, len(strings), FEATURE_CHUNK):
        P = torch.tensor(real_pauli_stack(strings[i:i + FEATURE_CHUNK]),
                         dtype=torch.float64)
        out.append(torch.einsum('bm,kmn,bn->bk', Xt, P, Xt).numpy())
        del P
    return np.concatenate(out, axis=1)


def evaluate(f_tr, f_te, ytr, yte):
    """Fit on the first k precomputed feature columns (already sliced by caller)."""
    sc = StandardScaler().fit(f_tr)
    clf = LogisticRegression(C=10.0, max_iter=2000, random_state=42)
    clf.fit(sc.transform(f_tr), ytr)
    ptr, pte = clf.predict(sc.transform(f_tr)), clf.predict(sc.transform(f_te))
    return dict(
        train_f1=f1_score(ytr, ptr, zero_division=0),
        test_f1=f1_score(yte, pte, zero_division=0),
        train_bal=balanced_accuracy_score(ytr, ptr),
        test_bal=balanced_accuracy_score(yte, pte),
    )


def run():
    os.makedirs('results', exist_ok=True)
    rows = []

    for ds, n_qubits, loader in DATASETS:
        all_strings = generate_pauli_strings(n_qubits)
        live = [s for s in all_strings if not is_dead_string(s)]
        grid = k_grid(len(live))
        print(f"\n=== {ds} N={n_qubits}  live={len(live)}  k grid={grid[:6]}...{grid[-2:]} ===",
              flush=True)

        for seed in SEEDS:
            Xtr, Xte, ytr, yte = loader(n_qubits, seed)
            n_train = len(Xtr)
            ranking, _, _ = generate_spectral_pauli_strings(Xtr, ytr, n_qubits)
            spectral = [s for s in ranking if not is_dead_string(s)]
            rng = np.random.default_rng(4000 + seed)
            rand = [live[i] for i in rng.permutation(len(live))]

            for arm, order in (('spectral', spectral), ('random_live', rand)):
                # Features for the whole ordered basis once; each k is a slice.
                F_tr = quadratic_features(Xtr, order)
                F_te = quadratic_features(Xte, order)
                for k in grid:
                    m = evaluate(F_tr[:, :k], F_te[:, :k], ytr, yte)
                    m.update(dataset=ds, n_qubits=n_qubits, seed=seed, arm=arm,
                             k=k, n_live=len(live), n_train=n_train,
                             k_over_n=k / n_train,
                             gap_f1=m['train_f1'] - m['test_f1'],
                             gap_bal=m['train_bal'] - m['test_bal'])
                    rows.append(m)
                del F_tr, F_te
            print(f"  seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('results/pruning_limits.csv', index=False)

    lines = ["Pruning limits and overfitting vs Pauli basis size",
             "=" * 80,
             f"Seeds: {len(SEEDS)}  |  metric: F1 (test), gap = train F1 - test F1",
             "",
             "(1) PRUNING LIMIT - smallest k reaching a share of full-basis test F1",
             "",
             f"  {'dataset':<9}{'n':>3}{'live':>7}{'full F1':>10}" +
             "".join(f"{'k@'+str(int(t*100))+'%':>10}" for t in RETENTION_TARGETS) +
             f"{'compression':>13}"]

    for (ds, nq), g in df[df.arm == 'spectral'].groupby(['dataset', 'n_qubits']):
        curve = g.groupby('k')['test_f1'].mean()
        n_live = int(g['n_live'].iloc[0])
        full = curve.loc[n_live]
        cells, best_k = [], None
        for t in RETENTION_TARGETS:
            ok = [k for k, v in curve.items() if v >= t * full]
            kk = min(ok) if ok else n_live
            cells.append(f"{kk:>10}")
            if t == 0.95:
                best_k = kk
        lines.append(f"  {ds:<9}{nq:>3}{n_live:>7}{full:>10.4f}" + "".join(cells) +
                     f"{n_live / max(best_k, 1):>12.1f}x")

    lines += ["", "(2) OVERFITTING - train/test gap as the basis grows",
              "    (k is also the parameter count: one w_j per string)", ""]
    for (ds, nq), g in df[df.arm == 'spectral'].groupby(['dataset', 'n_qubits']):
        n_train = int(g['n_train'].iloc[0])
        n_live = int(g['n_live'].iloc[0])
        lines.append(f"  {ds} N={nq}  (train samples = {n_train}, live basis = {n_live})")
        lines.append(f"    {'k':>7}{'k/N':>8}{'train F1':>11}{'test F1':>10}{'gap':>9}")
        curve = g.groupby('k')[['train_f1', 'test_f1', 'gap_f1', 'k_over_n']].mean()
        for k, r in curve.iterrows():
            mark = "  <-- k > N" if r['k_over_n'] > 1 else ""
            lines.append(f"    {int(k):>7}{r['k_over_n']:>8.2f}{r['train_f1']:>11.4f}"
                         f"{r['test_f1']:>10.4f}{r['gap_f1']:>9.4f}{mark}")
        peak = curve['test_f1'].idxmax()
        lines.append(f"    peak test F1 at k={int(peak)} "
                     f"({curve['test_f1'].max():.4f}); "
                     f"gap there {curve.loc[peak, 'gap_f1']:.4f}, "
                     f"gap at full basis {curve.loc[n_live, 'gap_f1']:.4f}")
        lines.append("")

    lines += ["(3) Does a worse ordering overfit faster? gap at the full basis:",
              f"  {'dataset':<9}{'n':>3}{'spectral':>11}{'random_live':>14}"]
    for (ds, nq), g in df.groupby(['dataset', 'n_qubits']):
        n_live = int(g['n_live'].iloc[0])
        s = g[(g.arm == 'spectral') & (g.k == n_live)]['gap_f1'].mean()
        r = g[(g.arm == 'random_live') & (g.k == n_live)]['gap_f1'].mean()
        lines.append(f"  {ds:<9}{nq:>3}{s:>11.4f}{r:>14.4f}")

    report = "\n".join(lines)
    print("\n" + report)
    with open('results/pruning_limits.txt', 'w') as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for (ds, nq), g in df[df.arm == 'spectral'].groupby(['dataset', 'n_qubits']):
        c = g.groupby('k')['test_f1'].mean()
        n_live = int(g['n_live'].iloc[0])
        ax.plot(c.index / n_live * 100, c.values / c.loc[n_live], 'o-',
                label=f'{ds} N={nq}')
    ax.axhline(0.95, ls='--', c='grey', label='95% of full basis')
    ax.set_xscale('log'); ax.set_xlabel('% of live basis retained')
    ax.set_ylabel('test F1 / full-basis F1')
    ax.set_title('How far can we prune?'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for (ds, nq), g in df[df.arm == 'spectral'].groupby(['dataset', 'n_qubits']):
        c = g.groupby('k')[['train_f1', 'test_f1']].mean()
        ax.plot(c.index, c['train_f1'], '--', alpha=0.6)
        ax.plot(c.index, c['test_f1'], 'o-', label=f'{ds} N={nq}')
    ax.set_xscale('log', base=2); ax.set_xlabel('Basis size k (= parameters)')
    ax.set_ylabel('F1 (dashed = train)')
    ax.set_title('Train vs test'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for arm, style in (('spectral', 'o-'), ('random_live', 's--')):
        g = df[df.arm == arm].groupby('k')['gap_f1'].mean()
        ax.plot(g.index, g.values, style, label=arm)
    ax.set_xscale('log', base=2); ax.set_xlabel('Basis size k')
    ax.set_ylabel('train F1 - test F1')
    ax.set_title('Overfitting gap'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/pruning_limits.png', dpi=150)
    print("\nSaved results/pruning_limits.{csv,txt,png}")


if __name__ == "__main__":
    run()
