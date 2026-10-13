import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from ..utils.data_loader import (
    ECOLI_CSV,
    ECOLI_FIRST_GENE_COLUMN,
    REPO_ROOT,
    load_ecoli_raw,
    select_topk_chi2,
)
from ..utils.paths import results_path
from ..utils.pauli_utils import get_pauli_tensor


def get_gene_names():
    """Gene column names, in the same column order as load_ecoli_raw() returns them."""
    return list(pd.read_csv(ECOLI_CSV, nrows=0).columns[ECOLI_FIRST_GENE_COLUMN:])


def get_selected_genes(n_qubits=4, test_size=0.3, random_state=42):
    """
    Re-runs the feature selection logic to get the exact names of the selected genes,
    in feature order.

    Reproduces the train-only chi2 selection of load_ecoli_split() for the same split
    arguments, i.e. the features experiment_ecoli_exact_ablation.py trains on when it
    writes results/pauli_importance_ranking.csv. This previously fit chi2 on the full
    dataset, which selects a different gene set (15 of 16 shared at random_state=42);
    because features are kept in column order, one substitution shifted 12 of the 16
    heatmap labels onto the wrong genes.
    """
    print("Loading Data to retrieve Gene Names...")
    X_genes, y = load_ecoli_raw()
    Xtr_raw, Xte_raw, y_train, _ = train_test_split(
        X_genes, y, test_size=test_size, random_state=random_state
    )
    _, _, selector = select_topk_chi2(Xtr_raw, Xte_raw, y_train, n_qubits=n_qubits)

    genes = get_gene_names()
    selected_genes = [genes[i] for i in selector.get_support(indices=True)]

    print(f"Selected Genes (N={n_qubits} / {2**n_qubits} dims): {selected_genes}")
    return selected_genes

def analyze_geometry():
    # 1. Get Gene Names
    gene_names = get_selected_genes()
    
    # 2. Load Top 32 Strings
    # We load from ranking, but we can also just use the constant list for consistency with last step
    # Let's load from CSV to get the SCORES (weights)
    print("Loading Pauli Importance Scores...")
    df_rank = pd.read_csv(os.path.join(REPO_ROOT, 'results', 'pauli_importance_ranking.csv'))


    # Take top 32
    top_32 = df_rank.head(32)
    
    # 3. Construct Aggregate Interaction Matrix
    # M_total = Sum( Score_s * |Real(P_s)| )
    # We take Absolute value of P to show "Connectivity Strength" regardless of +/- correlation
    # We take the Real part, matching the model's P_tensor (real_pauli_stack)
    
    dim = 16
    interaction_matrix = np.zeros((dim, dim))
    
    print("Computing Interaction Matrix...")
    for _, row in top_32.iterrows():
        p_str = row['String']
        score = row['Score']
        
        # Get Matrix (16x16)
        # Note: get_pauli_tensor returns complex numpy array
        P = get_pauli_tensor(p_str)
        P_real = np.real(P)
        
        # Add weighted absolute interaction
        # We perform element-wise abs to treat negative correlations as strong links
        interaction_matrix += score * np.abs(P_real)
        
    # Normalize diagonal vs off-diagonal?
    # Diagonal elements = variance/importance of single gene
    # Off-diagonal = interaction
    
    # 4. Plot Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(interaction_matrix, xticklabels=gene_names, yticklabels=gene_names, 
                cmap='viridis', annot=False)
    plt.title('Gene Interaction Intensity Map\n(Weighted Sum of Top 32 Pauli Strings)')
    plt.tight_layout()
    plt.savefig(results_path('gene_interaction_heatmap.png'))
    print("Saved heatmap to results/gene_interaction_heatmap.png")
    
    # 5. Extract Top Interactions
    # Find max off-diagonal elements
    analysis_lines = []
    analysis_lines.append("--- Top Gene Interactions ---")
    
    # Mask diagonal
    np.fill_diagonal(interaction_matrix, 0)
    
    # Get indices of top interactions
    # interaction matrix is symmetric
    flat_indices = np.argsort(interaction_matrix.ravel())[::-1]
    
    seen_pairs = set()
    count = 0
    for idx_flat in flat_indices:
        i, j = np.unravel_index(idx_flat, (dim, dim))
        if i >= j: continue # Symmetry or diagonal
        
        pair = tuple(sorted((i, j)))
        if pair in seen_pairs: continue
        seen_pairs.add(pair)
        
        val = interaction_matrix[i, j]
        if val < 0.1: break # Threshold
        
        gene_a = gene_names[i]
        gene_b = gene_names[j]
        line = f"Rank {count+1}: {gene_a} <-> {gene_b} (Strength: {val:.2f})"
        analysis_lines.append(line)
        print(line)
        
        count += 1
        if count >= 15: break
        
    with open(results_path('gene_interaction_report.txt'), 'w') as f:
        f.write('\n'.join(analysis_lines))

if __name__ == "__main__":
    analyze_geometry()
