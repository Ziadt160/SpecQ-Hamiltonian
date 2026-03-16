import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from src.generators.spectral_pauli_generator import generate_spectral_pauli_strings
from src.models.exact_sim_classifier import ExactSIMClassifier
from src.utils.pauli_utils import generate_pauli_strings
import os

def load_ecoli_n6():
    """
    Loads EColi and selects top 64 genes (N=6).
    """
    print("Loading E. Coli dataset...")
    df = pd.read_csv(r'd:\Evoth Labs\SIM-Flipped Models\data\EColi_Merged_df.csv')
    df = df.dropna(subset=['CTZ'])
    y = df['CTZ'].apply(lambda x: 1 if x == 'R' else 0).values
    
    # Identify Gene Columns (assume after CIP based on previous knowledge)
    cols = list(df.columns)
    try:
        start_idx = cols.index('CIP') + 1
    except ValueError:
        start_idx = 15
    gene_cols = cols[start_idx:]
    X_genes = df[gene_cols].values
    
    # Select Top 64
    n_qubits = 6
    target_dim = 2**n_qubits # 64
    print(f"Selecting Top {target_dim} Genes...")
    selector = SelectKBest(chi2, k=target_dim)
    X_reduced = selector.fit_transform(X_genes, y)
    
    # Normalize
    norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X_reduced / norms
    
    return X_norm, y

def run_ecoli_spectral_experiment():
    n_qubits = 6
    X, y = load_ecoli_n6()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. Spectral Selection
    print("Running Spectral Pauli Generator (N=6)...")
    # This might take a moment for 4096 interactions
    spectral_strings, coefs, mags = generate_spectral_pauli_strings(X_train, y_train, n_qubits)
    
    # Select Top 256 (6.25% of basis)
    k = 256
    top_k_strings = spectral_strings[:k]
    
    # Random Baseline
    import random
    all_strings = generate_pauli_strings(n_qubits)
    random_strings = random.sample(all_strings, k)
    
    # Prepare Data
    X_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)
    
    def train_model(name, p_strings):
        print(f"\nTraining {name} with {len(p_strings)} Pauli strings...")
        model = ExactSIMClassifier(n_qubits=n_qubits, n_layers=2, pauli_strings=p_strings)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        # Train
        epochs = 30 # Reduced for speed, sufficient for convergence usually
        for ep in range(epochs):
            optimizer.zero_grad()
            out = model(X_train_t)
            loss = criterion(out, y_train_t)
            loss.backward()
            optimizer.step()
            if ep % 10 == 0:
                print(f"  Ep {ep}: loss={loss.item():.4f}")
        
        # Evaluate
        with torch.no_grad():
            preds = (model(X_test_t) > 0.5).float()
            acc = accuracy_score(y_test_t, preds)
        print(f"  -> {name} Acc: {acc:.4f}")
        return acc

    # Comparison
    # Note: Full basis (4096) might be too slow for this quick turnaround. 
    # But let's try Spectral vs Random.
    # If possible, run full, but it's 16x more terms.
    
    acc_spec = train_model("Spectral Top-256", top_k_strings)
    acc_rand = train_model("Random Top-256", random_strings)
    
    # Save Results
    with open('../results/ecoli_spectral_results.txt', 'w') as f:
        f.write(f"E. Coli N=6 Experiment\n")
        f.write(f"Spectral Top-{k} Acc: {acc_spec:.4f}\n")
        f.write(f"Random Top-{k} Acc: {acc_rand:.4f}\n")
        f.write(f"Improvement: {acc_spec - acc_rand:.4f}\n")

if __name__ == "__main__":
    run_ecoli_spectral_experiment()
