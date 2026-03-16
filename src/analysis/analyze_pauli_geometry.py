import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from ..utils.pauli_utils import get_pauli_tensor
from ..utils.data_loader import load_ecoli_reduced as load_ecoli_data

def get_selected_genes():
    """
    Re-runs the feature selection logic to get the exact names of the 16 selected genes.
    """
    print("Loading Data to retrieve Gene Names...")
    try:
        df = pd.read_csv(r'd:\Evoth Labs\SIM-Flipped Models\data\EColi_Merged_df.csv')
    except:
        df = pd.read_csv('data/EColi_Merged_df.csv')
        
    df = df.dropna(subset=['CTZ'])
    y = df['CTZ'].apply(lambda x: 1 if x == 'R' else 0).values
    
    # Feature columns
    cols = list(df.columns)
    try: start_idx = cols.index('CIP') + 1
    except: start_idx = 15
    genes = df.columns[start_idx:]
    X_genes = df.iloc[:, start_idx:].values
    
    # Run Selection
    selector = SelectKBest(chi2, k=16)
    selector.fit(X_genes, y)
    
    selected_indices = selector.get_support(indices=True)
    selected_genes = [genes[i] for i in selected_indices]
    
    print(f"Selected Genes (N=4 / 16 dims): {selected_genes}")
    return selected_genes

def analyze_geometry():
    # 1. Get Gene Names
    gene_names = get_selected_genes()
    
    # 2. Load Top 32 Strings
    # We load from ranking, but we can also just use the constant list for consistency with last step
    # Let's load from CSV to get the SCORES (weights)
    print("Loading Pauli Importance Scores...")
    try:
        df_rank = pd.read_csv(r'd:\Evoth Labs\SIM-Flipped Models\results\pauli_importance_ranking.csv')
    except:
        df_rank = pd.read_csv('results/pauli_importance_ranking.csv')
        
    # Take top 32
    top_32 = df_rank.head(32)
    
    # 3. Construct Aggregate Interaction Matrix
    # M_total = Sum( Score_s * |Real(P_s)| )
    # We take Absolute value of P to show "Connectivity Strength" regardless of +/- correlation
    # We take Real part because pytorch cast to double discards imaginary
    
    dim = 16
    interaction_matrix = np.zeros((dim, dim))
    
    print("Computing Interaction Matrix...")
    for idx, row in top_32.iterrows():
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
    plt.savefig('results/gene_interaction_heatmap.png')
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
        
    with open('results/gene_interaction_report.txt', 'w') as f:
        f.write('\n'.join(analysis_lines))

if __name__ == "__main__":
    analyze_geometry()
