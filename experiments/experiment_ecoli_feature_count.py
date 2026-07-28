"""
How much of the E. Coli result is an artifact of reducing to 2^n genes?

Reducing the 17,198-gene matrix to 2^4 = 16 chi2-selected genes leaves the
majority of samples as all-zero vectors. For a quadratic model
phi_j(x) = x^T P_j x an all-zero input produces the *same* feature vector for
every such sample (phi_j(0 + b) = b^T P_j b), so the model cannot tell them
apart -- they all receive one constant prediction. With a strongly imbalanced
label set, that constant prediction is the majority class, and accuracy looks
high for structural rather than discriminative reasons.

This sweep quantifies the tradeoff: as the retained gene count k grows, what
happens to the blank-row fraction, and does discriminative performance (F1,
balanced accuracy) actually improve?

All selection is fit on the training split only.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.utils.data_loader import load_ecoli_raw, row_normalize
from src.utils.pauli_utils import generate_pauli_strings
from src.models.sim_classifier import SIMClassifier

K_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]
# Gene counts that are an exact power of two can also feed the Pauli quadratic
# model (dim must equal 2^n). Beyond N=6 the basis (4^n strings) is too large.
PAULI_DIMS = {16: 4, 64: 6}


def evaluate_k(X_genes, y, k, test_size=0.3, random_state=42):
    Xtr_raw, Xte_raw, y_train, y_test = train_test_split(
        X_genes, y, test_size=test_size, random_state=random_state
    )
    selector = SelectKBest(chi2, k=k).fit(Xtr_raw, y_train)
    X_train = row_normalize(selector.transform(Xtr_raw))
    X_test = row_normalize(selector.transform(Xte_raw))

    blank_tr = float((np.linalg.norm(X_train, axis=1) == 0).mean())
    blank_te = float((np.linalg.norm(X_test, axis=1) == 0).mean())
    majority = float(max(y_test.mean(), 1 - y_test.mean()))

    row = {
        'k_genes': k,
        'blank_train': blank_tr,
        'blank_test': blank_te,
        'majority_baseline': majority,
    }

    # Linear baseline on the selected genes.
    lr = LogisticRegression(max_iter=2000, C=10.0).fit(X_train, y_train)
    p = lr.predict(X_test)
    row.update(
        lr_accuracy=accuracy_score(y_test, p),
        lr_balanced=balanced_accuracy_score(y_test, p),
        lr_f1=f1_score(y_test, p, zero_division=0),
    )

    # Quadratic Pauli-feature model, where the dimension permits it.
    if k in PAULI_DIMS:
        n_qubits = PAULI_DIMS[k]
        strings = generate_pauli_strings(n_qubits)
        sim = SIMClassifier(pauli_strings=strings, C=10.0, random_state=42)
        sim.fit(X_train, y_train)
        p = sim.predict(X_test)
        row.update(
            sim_accuracy=accuracy_score(y_test, p),
            sim_balanced=balanced_accuracy_score(y_test, p),
            sim_f1=f1_score(y_test, p, zero_division=0),
            sim_basis=len(strings),
        )
    else:
        row.update(sim_accuracy=np.nan, sim_balanced=np.nan,
                   sim_f1=np.nan, sim_basis=np.nan)
    return row


def run_feature_count_sweep():
    os.makedirs('results', exist_ok=True)

    X_genes, y = load_ecoli_raw()
    nonzero_genes = int((X_genes.sum(axis=0) > 0).sum())
    print(f"E. Coli: {X_genes.shape[0]} samples x {X_genes.shape[1]} genes "
          f"({nonzero_genes} genes non-zero somewhere)")
    print(f"Label balance: {y.mean():.1%} resistant\n")

    rows = []
    for k in K_VALUES:
        print(f"--- k = {k} genes ---", flush=True)
        row = evaluate_k(X_genes, y, k)
        rows.append(row)
        msg = (f"  blank rows: train {row['blank_train']:.1%} / test {row['blank_test']:.1%}"
               f" | majority {row['majority_baseline']:.4f}"
               f" | LR acc {row['lr_accuracy']:.4f} bal {row['lr_balanced']:.4f}"
               f" F1 {row['lr_f1']:.4f}")
        if not np.isnan(row['sim_accuracy']):
            msg += (f"\n      SIM (N={PAULI_DIMS[k]}, {int(row['sim_basis'])} paulis)"
                    f" acc {row['sim_accuracy']:.4f} bal {row['sim_balanced']:.4f}"
                    f" F1 {row['sim_f1']:.4f}")
        print(msg, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('results/ecoli_feature_count.csv', index=False)

    lines = [
        "E. Coli feature-count sweep (chi2 selection fit on train split only)",
        "=" * 76,
        f"{X_genes.shape[0]} samples, {X_genes.shape[1]} genes, {y.mean():.1%} resistant",
        "",
        f"{'genes':>7}{'blank tr':>10}{'blank te':>10}{'major':>9}"
        f"{'LR acc':>9}{'LR bal':>9}{'LR F1':>8}{'SIM acc':>9}{'SIM F1':>9}",
    ]
    for _, r in df.iterrows():
        sim_a = "  -  " if np.isnan(r['sim_accuracy']) else f"{r['sim_accuracy']:.4f}"
        sim_f = "  -  " if np.isnan(r['sim_f1']) else f"{r['sim_f1']:.4f}"
        lines.append(
            f"{int(r['k_genes']):>7}{r['blank_train']:>9.1%}{r['blank_test']:>10.1%}"
            f"{r['majority_baseline']:>9.4f}{r['lr_accuracy']:>9.4f}"
            f"{r['lr_balanced']:>9.4f}{r['lr_f1']:>8.4f}{sim_a:>9}{sim_f:>9}"
        )
    lines += [
        "",
        "blank tr/te = fraction of samples that are all-zero after selection.",
        "Those samples are indistinguishable to any model of the form f(x^T P x),",
        "so they all receive one constant prediction.",
        "",
        "Accuracy should be read against the majority-class column, not against 0.5.",
        "F1 and balanced accuracy are the informative metrics on this label split.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open('results/ecoli_feature_count.txt', 'w') as f:
        f.write(report + "\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(df['k_genes'], df['blank_test'] * 100, 'o-', color='crimson')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Genes retained (chi2)')
    ax.set_ylabel('All-zero test samples (%)')
    ax.set_title('Blank rows vanish as the gene budget grows')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df['k_genes'], df['lr_f1'], 'o-', label='Logistic regression F1', color='#4c72b0')
    ax.plot(df['k_genes'], df['lr_balanced'], 's-', label='Logistic regression bal-acc',
            color='#55a868')
    sim = df.dropna(subset=['sim_f1'])
    ax.plot(sim['k_genes'], sim['sim_f1'], '^', markersize=11,
            label='SIM (Pauli quadratic) F1', color='#dd8452')
    ax.axhline(0.0, color='grey', lw=0.5)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Genes retained (chi2)')
    ax.set_ylabel('Score')
    ax.set_title('Discriminative performance vs gene budget')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/ecoli_feature_count.png', dpi=150)
    print("\nSaved results/ecoli_feature_count.{csv,txt,png}")


if __name__ == "__main__":
    run_feature_count_sweep()
