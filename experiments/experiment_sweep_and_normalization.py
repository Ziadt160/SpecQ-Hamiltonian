"""
Two settings where this repository departs from Tiblias et al., tested directly.

(1) Pauli-string count.
    The paper's tuned SIM uses p = 1000 on every dataset (Table 6) and Sec. C.3
    reports accuracy "steadily increases and eventually plateaus at 1000",
    speculating that more would help further. This repository reports the
    opposite -- a sweet spot near top-128 with diminishing returns after.
    The two are not directly comparable: at n=4 the *entire* basis is 256
    strings, so p=128 is half of it, whereas the paper works at n=9-10 where
    p=1000 is under 0.4% of 4^n. This sweep measures p as a *fraction of the
    live basis* so the two regimes can be compared on the same axis.

(2) Input normalization.
    The paper uses raw embeddings -- "no further pre-processing or
    dimensionality reduction is performed". This repository L2-normalizes every
    sample to unit norm, commented "Quantum State Normalization". But in a
    flipped model the input is an *observable*, not a state: it never needs unit
    norm, and since alpha_j = x^T P_j x is quadratic, normalizing discards
    per-sample magnitude. This ablation measures what that costs.

Both use SIMClassifier, which experiment_circuit_ablation.py established is the
equivalent hypothesis class to the full quantum model, at a fraction of the cost.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.generators.spectral_pauli_generator import (
    generate_spectral_pauli_strings, is_dead_string, count_dead_strings)
from src.models.sim_classifier import SIMClassifier
from src.utils.data_loader import (
    load_ecoli_raw, row_normalize, _fetch_20newsgroups)
from src.utils.pauli_utils import generate_pauli_strings

SEEDS = [0, 1, 2]
FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]


def score(basis, Xtr, ytr, Xte, yte):
    sim = SIMClassifier(pauli_strings=basis, C=10.0, random_state=42)
    sim.fit(Xtr, ytr)
    p = sim.predict(Xte)
    return (accuracy_score(yte, p), balanced_accuracy_score(yte, p),
            f1_score(yte, p, zero_division=0))


# --------------------------------------------------------------------------
# (1) Pauli count as a fraction of the live basis
# --------------------------------------------------------------------------

def ecoli_split(n_qubits, seed, normalize=True):
    from sklearn.feature_selection import SelectKBest, chi2
    X_genes, y = load_ecoli_raw()
    a, b, ytr, yte = train_test_split(X_genes, y, test_size=0.3, random_state=42 + seed)
    sel = SelectKBest(chi2, k=2 ** n_qubits).fit(a, ytr)
    Xtr, Xte = sel.transform(a), sel.transform(b)
    if normalize:
        Xtr, Xte = row_normalize(Xtr), row_normalize(Xte)
    return Xtr, Xte, ytr, yte


def run_count_sweep():
    rows = []
    for n_qubits in (4, 6):
        n_live = 4 ** n_qubits - count_dead_strings(n_qubits)
        print(f"\n--- Pauli count sweep, N={n_qubits} "
              f"(live basis {n_live}/{4**n_qubits}) ---", flush=True)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = ecoli_split(n_qubits, seed)
            ranking, _, _ = generate_spectral_pauli_strings(Xtr, ytr, n_qubits)
            live_ranking = [s for s in ranking if not is_dead_string(s)]
            for frac in FRACTIONS:
                k = max(1, int(round(frac * n_live)))
                acc, bal, f1 = score(live_ranking[:k], Xtr, ytr, Xte, yte)
                rows.append(dict(n_qubits=n_qubits, seed=seed, fraction=frac,
                                 k=k, n_live=n_live,
                                 accuracy=acc, balanced=bal, f1=f1))
            print(f"  seed {seed} done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# (2) Normalization ablation
# --------------------------------------------------------------------------

def news_variants(n_qubits, seed):
    """20 Newsgroups projected to 2^n dims, with and without L2 row normalization."""
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    ng = _fetch_20newsgroups()
    dtr, dte, ytr, yte = train_test_split(ng.data, ng.target, test_size=0.3,
                                          random_state=42 + seed)
    vec = TfidfVectorizer(stop_words='english', max_features=5000).fit(dtr)
    pca = PCA(n_components=2 ** n_qubits, random_state=42).fit(vec.transform(dtr).toarray())
    sc = StandardScaler().fit(pca.transform(vec.transform(dtr).toarray()))

    def proj(docs):
        return sc.transform(pca.transform(vec.transform(docs).toarray()))

    raw_tr, raw_te = proj(dtr), proj(dte)
    return {
        'raw (paper)': (raw_tr, raw_te),
        'L2 row-norm (repo)': (row_normalize(raw_tr.copy()), row_normalize(raw_te.copy())),
    }, ytr, yte


def run_normalization_ablation():
    rows = []
    for n_qubits in (4,):
        print(f"\n--- Normalization ablation, N={n_qubits} ---", flush=True)
        for seed in SEEDS:
            # 20 Newsgroups
            variants, ytr, yte = news_variants(n_qubits, seed)
            for name, (Xtr, Xte) in variants.items():
                ranking, _, _ = generate_spectral_pauli_strings(Xtr, ytr, n_qubits)
                acc, bal, f1 = score(ranking[:32], Xtr, ytr, Xte, yte)
                rows.append(dict(dataset='20News', n_qubits=n_qubits, seed=seed,
                                 preprocessing=name, accuracy=acc, balanced=bal, f1=f1))
            # E. Coli
            for name, norm in [('raw (paper)', False), ('L2 row-norm (repo)', True)]:
                Xtr, Xte, etr, ete = ecoli_split(n_qubits, seed, normalize=norm)
                ranking, _, _ = generate_spectral_pauli_strings(Xtr, etr, n_qubits)
                acc, bal, f1 = score(ranking[:32], Xtr, etr, Xte, ete)
                rows.append(dict(dataset='EColi', n_qubits=n_qubits, seed=seed,
                                 preprocessing=name, accuracy=acc, balanced=bal, f1=f1))
            print(f"  seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def main():
    os.makedirs('results', exist_ok=True)
    dfc = run_count_sweep()
    dfn = run_normalization_ablation()
    dfc.to_csv('results/pauli_count_sweep.csv', index=False)
    dfn.to_csv('results/normalization_ablation.csv', index=False)

    lines = ["Pauli-count sweep and normalization ablation",
             "=" * 74, "",
             "(1) Accuracy vs basis size, as a FRACTION of the live basis.",
             "    The paper's p=1000 at n=10 is 0.19% of the live basis;",
             "    this repo's top-128 at n=4 is 94% of it. Same axis, finally.", "",
             f"  {'N':>3}{'fraction':>10}{'k':>7}{'accuracy':>11}{'bal-acc':>10}{'F1':>9}"]
    for (nq, frac), g in dfc.groupby(['n_qubits', 'fraction']):
        lines.append(f"  {nq:>3}{frac:>9.0%}{int(g['k'].mean()):>7}"
                     f"{g['accuracy'].mean():>11.4f}{g['balanced'].mean():>10.4f}"
                     f"{g['f1'].mean():>9.4f}")

    lines += ["", "(2) Input normalization: paper (raw) vs repository (L2 row-norm).", "",
              f"  {'dataset':<9}{'preprocessing':<22}{'accuracy':>11}{'bal-acc':>10}{'F1':>9}"]
    for (ds, pre), g in dfn.groupby(['dataset', 'preprocessing']):
        lines.append(f"  {ds:<9}{pre:<22}{g['accuracy'].mean():>11.4f}"
                     f"{g['balanced'].mean():>10.4f}{g['f1'].mean():>9.4f}")

    for ds in dfn['dataset'].unique():
        sub = dfn[dfn.dataset == ds]
        raw = sub[sub.preprocessing == 'raw (paper)']['f1'].mean()
        l2 = sub[sub.preprocessing == 'L2 row-norm (repo)']['f1'].mean()
        verdict = 'hurts' if raw > l2 else 'helps'
        lines.append(f"  -> {ds}: L2 row-normalization {verdict}, "
                     f"F1 {l2 - raw:+.4f} vs the paper's raw inputs")

    report = "\n".join(lines)
    print("\n" + report)
    with open('results/sweep_and_normalization.txt', 'w') as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for nq in sorted(dfc['n_qubits'].unique()):
        g = dfc[dfc.n_qubits == nq].groupby('fraction')[['f1']].mean()
        ax.plot(g.index * 100, g['f1'], 'o-', label=f'N={nq}')
    ax.set_xscale('log')
    ax.set_xlabel('Basis size (% of live Pauli basis)')
    ax.set_ylabel('F1'); ax.set_title('Does more Pauli strings keep helping?')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    pre = list(dfn['preprocessing'].unique())
    xs = np.arange(len(pre)); width = 0.35
    for i, ds in enumerate(dfn['dataset'].unique()):
        sub = dfn[dfn.dataset == ds]
        means = [sub[sub.preprocessing == p]['f1'].mean() for p in pre]
        errs = [sub[sub.preprocessing == p]['f1'].std() for p in pre]
        ax.bar(xs + i * width, means, width, yerr=errs, capsize=4, label=ds)
    ax.set_xticks(xs + width / 2); ax.set_xticklabels(pre, fontsize=9)
    ax.set_ylabel('F1'); ax.set_title('Cost of L2 row-normalization')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/sweep_and_normalization.png', dpi=150)
    print("\nSaved results/{pauli_count_sweep,normalization_ablation}.csv "
          "and results/sweep_and_normalization.{txt,png}")


if __name__ == "__main__":
    main()
