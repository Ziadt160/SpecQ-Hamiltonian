import sys
import os
print("Starting script execution...", flush=True)
# Hack to resolve mixed imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.models.nisq_sim_classifier import NISQSIMClassifier
from src.generators.spectral_pauli_generator import generate_spectral_pauli_strings
from src.utils.pauli_utils import generate_pauli_strings
from src.utils.data_loader import load_ecoli_reduced as load_ecoli_n4
import random

def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=150):
    opt = optim.Adam(model.parameters(), lr=0.01)
    crt = nn.BCELoss()
    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    X_te = torch.tensor(X_test, dtype=torch.float64)
    
    for _ in range(epochs):
        opt.zero_grad()
        loss = crt(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        
    with torch.no_grad():
        pred_te = (model(X_te) > 0.5).float().detach().numpy()
        acc_te = accuracy_score(y_test, pred_te)
        
    return acc_te

def run_nisq_experiment(n_seeds=3):
    print("Loading E. Coli (N=4)...")
    X, y = load_ecoli_n4()
    
    # We stop at 64 because NISQ simulation (density matrix) is slow
    k_values = [2, 4, 8, 16, 32, 64] 
    
    acc_spec = np.zeros((n_seeds, len(k_values)))
    acc_rand = np.zeros((n_seeds, len(k_values)))
    
    print(f"Running NISQ Noise Sweep with {n_seeds} seeds...")
    
    for seed in range(n_seeds):
        print(f"--- Seed {seed+1}/{n_seeds} ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42+seed
        )
        
        spectral_ranking, _, _ = generate_spectral_pauli_strings(X_train, y_train, 4)
        full_basis = generate_pauli_strings(4)
        
        for i, k in enumerate(k_values):
            print(f"  > Basis Size k={k} ({i+1}/{len(k_values)})", flush=True)
            
            # Spectral 
            spec_subset = spectral_ranking[:k]
            model_spec = NISQSIMClassifier(4, pauli_strings=spec_subset, device_name='default.mixed')
            acc_spec[seed, i] = train_and_evaluate(model_spec, X_train, y_train, X_test, y_test)
            
            # Random
            random.seed(seed + k)
            rand_subset = random.sample(full_basis, k)
            model_rand = NISQSIMClassifier(4, pauli_strings=rand_subset, device_name='default.mixed')
            acc_rand[seed, i] = train_and_evaluate(model_rand, X_train, y_train, X_test, y_test)

    # Plot
    mean_spec = np.mean(acc_spec, axis=0)
    mean_rand = np.mean(acc_rand, axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, mean_spec, 'o-', label='Spectral Pruned (NISQ)', color='green')
    plt.plot(k_values, mean_rand, 's-', label='Random Basis (NISQ)', color='red')
    plt.xlabel('Basis Size (k)')
    plt.ylabel('Test Accuracy (Noisy)')
    plt.title('Performance under Realistic NISQ Noise')
    plt.legend()
    plt.xscale('log', base=2)
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/nisq_robustness.png')
    print("Saved plot to results/nisq_robustness.png")

if __name__ == "__main__":
    run_nisq_experiment()
