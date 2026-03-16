import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from src.utils.pauli_utils import generate_pauli_strings, classify_pauli_string, get_pauli_tensor
from src.models.sim_classifier import SIMClassifier
from src.models.exact_sim_classifier import ExactSIMClassifier
from src.experiment_ecoli_exact_ablation import load_ecoli_n4, calculate_pauli_importance
from src.analyze_pauli_geometry import get_selected_genes

def train_and_rank(seed, X, y):
    print(f"\n--- Run Seed {seed} ---")
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    
    # Train
    # Using slightly fewer epochs (150) for speed in validation loop, 
    # assuming convergence is fast enough for feature importance stability.
    # But for rigor, let's stick to 200.
    epochs = 200
    
    # Convert to Tensor (Double prec)
    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    
    model = ExactSIMClassifier(n_qubits=4, n_layers=3) # Random init inside
    # Set seed for torch/numpy inside model init if needed, but we want random init?
    # Actually, we want to see if the *data signal* is strong enough to overcome random init and random split.
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()
        
    # Importance
    # Use full X for importance? Or X_train? 
    # Paper/Standard is usually Test or Full. Let's use X_train consistent with previous step.
    imp_scores = calculate_pauli_importance(model, X_train)
    
    return model.pauli_strings, imp_scores

def run_stability_check():
    # 0. Load Data & Genes
    X, y = load_ecoli_n4()
    gene_names = get_selected_genes()
    dim = 16
    
    seeds = [0, 1, 2, 3, 4]
    
    # Accumulators
    pair_counts = {} # (GeneA, GeneB) -> count
    total_interaction_matrix = np.zeros((dim, dim))
    
    for seed in seeds:
        # 1. Train & Rank
        p_strings, scores = train_and_rank(seed, X, y)
        
        # 2. Get Top 32
        sorted_indices = np.argsort(scores)[::-1].copy()
        top_k_indices = sorted_indices[:32]
        
        # 3. Build Interaction Matrix for this Seed
        seed_matrix = np.zeros((dim, dim))
        
        for idx in top_k_indices:
            p_str = p_strings[idx]
            score = scores[idx]
            P = get_pauli_tensor(p_str)
            P_real = np.real(P)
            seed_matrix += score * np.abs(P_real)
            
        total_interaction_matrix += seed_matrix
        
        # 4. Count Pairs in Top 20 of this seed
        # Mask diagonal
        off_diag = seed_matrix.copy()
        np.fill_diagonal(off_diag, 0)
        
        # Get Top 20 pairs
        flat_indices = np.argsort(off_diag.ravel())[::-1]
        
        seen_in_seed = set()
        count = 0
        for idx_flat in flat_indices:
            i, j = np.unravel_index(idx_flat, (dim, dim))
            if i >= j: continue
            
            val = off_diag[i, j]
            if val < 0.05: break
            
            pair = tuple(sorted((gene_names[i], gene_names[j])))
            if pair in seen_in_seed: continue
            seen_in_seed.add(pair)
            
            # Add to global counter
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            count += 1
            if count >= 20: break
            
    # --- Final Analysis ---
    
    # 1. Stability Report
    print("\n--- Stability Validation Report (5 Folds) ---")
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    
    report_lines = []
    report_lines.append(f"Stability Validation over {len(seeds)} random seeds.")
    report_lines.append("Gene Pair | Frequency (Max 5)")
    report_lines.append("-" * 35)
    
    for pair, freq in sorted_pairs:
        line = f"{pair[0]} <-> {pair[1]} | {freq}/5"
        report_lines.append(line)
        print(line)
        
    with open('results/stability_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
        
    # 2. Consensus Heatmap
    mean_matrix = total_interaction_matrix / len(seeds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(mean_matrix, xticklabels=gene_names, yticklabels=gene_names, 
                cmap='viridis', annot=False)
    plt.title('Consensus Gene Interaction Map\n(Averaged over 5 Random Splits)')
    plt.tight_layout()
    plt.savefig('results/stability_heatmap.png')
    print("\nSaved Consensus Heatmap to 'results/stability_heatmap.png'")

if __name__ == "__main__":
    run_stability_check()
