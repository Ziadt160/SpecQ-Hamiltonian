import sys
import os
print("Starting script execution...", flush=True)
# Hack to resolve mixed imports (some use src., some use local)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from exact_sim_classifier import ExactSIMClassifier
from spectral_pauli_generator import generate_spectral_pauli_strings
from pauli_utils import generate_pauli_strings
from experiment_ecoli_exact import load_ecoli_n4

def train_model(model, X_train, y_train, epochs=200, lr=0.01):
    opt = optim.Adam(model.parameters(), lr=lr)
    crt = nn.BCELoss()
    X_t = torch.tensor(X_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    
    for _ in range(epochs):
        opt.zero_grad()
        loss = crt(model(X_t), y_t)
        loss.backward()
        opt.step()
    return model

def run_small_data_experiment(n_seeds=5):
    print("Loading E. Coli (N=4)...")
    X, y = load_ecoli_n4()
    
    # Data fractions
    fractions = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    
    results_spec = np.zeros((n_seeds, len(fractions)))
    results_full = np.zeros((n_seeds, len(fractions)))
    
    print(f"Running Small Data Stress Test with {n_seeds} seeds...")
    
    # We want to test on a FIXED test set, but vary training set size.
    # So we first split out a large Test Set.
    # If dataset is small, this is tricky. E. Coli is ~450 samples?
    # Let's verify size. Assuming small.
    # We'll use 30% for Test, 70% for Train (Pool).
    # Then subsample from Train Pool.
    
    for seed in range(n_seeds):
        print(f"--- Seed {seed+1}/{n_seeds} ---")
        X_train_pool, X_test, y_train_pool, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42+seed
        )
        
        # Fixed Test Tensor
        X_test_t = torch.tensor(X_test, dtype=torch.float64)
        
        for i, frac in enumerate(fractions):
            print(f"  > Data Fraction {frac*100:.0f}% ({i+1}/{len(fractions)})", flush=True)
            # Subsample Training Data
            if frac == 1.0:
                X_train_sub, y_train_sub = X_train_pool, y_train_pool
            else:
                # Stratified subsample?
                # Simple random sample
                n_sub = int(len(X_train_pool) * frac)
                # Ensure at least minimal samples
                if n_sub < 10: n_sub = 10
                
                # Use train_test_split to get random subset
                X_train_sub, _, y_train_sub, _ = train_test_split(
                    X_train_pool, y_train_pool, train_size=n_sub, random_state=seed, stratify=y_train_pool
                )
            
            # 1. Spectral Selection (Must use only available small data!)
            # This is key: Spectral method uses analytical stats, should be robust.
            spec_basis, _, _ = generate_spectral_pauli_strings(X_train_sub, y_train_sub, 4)
            spec_basis_top32 = spec_basis[:32]
            
            # 2. Full Basis (256)
            full_basis = generate_pauli_strings(4)
            
            # Train Spectral
            model_spec = ExactSIMClassifier(4, pauli_strings=spec_basis_top32)
            train_model(model_spec, X_train_sub, y_train_sub)
            
            # Train Full
            model_full = ExactSIMClassifier(4, pauli_strings=full_basis)
            train_model(model_full, X_train_sub, y_train_sub)
            
            # Evaluate
            pred_s = (model_spec(X_test_t) > 0.5).float().detach().numpy()
            results_spec[seed, i] = accuracy_score(y_test, pred_s)
            
            pred_f = (model_full(X_test_t) > 0.5).float().detach().numpy()
            results_full[seed, i] = accuracy_score(y_test, pred_f)

    # Stats
    mean_spec = np.mean(results_spec, axis=0)
    std_spec = np.std(results_spec, axis=0)
    mean_full = np.mean(results_full, axis=0)
    std_full = np.std(results_full, axis=0)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fractions, mean_spec, 'o-', label='Spectral (Top-32)', color='green')
    plt.fill_between(fractions, mean_spec - std_spec, mean_spec + std_spec, color='green', alpha=0.15)
    
    plt.plot(fractions, mean_full, 's-', label='Full Basis (256)', color='red')
    plt.fill_between(fractions, mean_full - std_full, mean_full + std_full, color='red', alpha=0.15)
    
    plt.xlabel('Fraction of Training Data Used')
    plt.ylabel('Test Accuracy')
    plt.title('Small Data Stress Test: Spectral vs Full')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/small_data_stress.png')
    print("Saved plot to results/small_data_stress.png")

if __name__ == "__main__":
    run_small_data_experiment()
