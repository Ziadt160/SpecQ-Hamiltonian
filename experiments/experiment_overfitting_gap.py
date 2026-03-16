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
import random

def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=200):
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
        # Train Acc
        pred_tr = (model(X_tr) > 0.5).float().detach().numpy()
        acc_tr = accuracy_score(y_train, pred_tr)
        
        # Test Acc
        pred_te = (model(X_te) > 0.5).float().detach().numpy()
        acc_te = accuracy_score(y_test, pred_te)
        
    return acc_tr, acc_te

def run_overfitting_gap_experiment(n_seeds=5):
    print("Loading E. Coli (N=4)...")
    X, y = load_ecoli_n4()
    
    # Basis sizes to sweep
    k_values = [2, 4, 8, 16, 32, 64, 128, 256] 
    
    # Storage: [seed, k]
    gap_spec = np.zeros((n_seeds, len(k_values)))
    gap_rand = np.zeros((n_seeds, len(k_values)))
    
    train_acc_spec = np.zeros((n_seeds, len(k_values)))
    test_acc_spec = np.zeros((n_seeds, len(k_values)))
    
    train_acc_rand = np.zeros((n_seeds, len(k_values)))
    test_acc_rand = np.zeros((n_seeds, len(k_values)))

    print(f"Running Overfitting Gap Analysis with {n_seeds} seeds...")
    
    for seed in range(n_seeds):
        print(f"--- Seed {seed+1}/{n_seeds} ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42+seed
        )
        
        # 1. Generate Spectral Basis (Full ranking)
        spectral_ranking, _, _ = generate_spectral_pauli_strings(X_train, y_train, 4)
        full_basis = generate_pauli_strings(4) # Normalized order or just standard
        
        for i, k in enumerate(k_values):
            print(f"  > Basis Size k={k} ({i+1}/{len(k_values)})", flush=True)
            # Spectral Selection
            spec_subset = spectral_ranking[:k]
            
            # Random Selection
            # Use same seed?
            random.seed(seed + k)
            rand_subset = random.sample(full_basis, k)
            
            # Train Spectral
            model_spec = ExactSIMClassifier(4, pauli_strings=spec_subset)
            tr_s, te_s = train_and_evaluate(model_spec, X_train, y_train, X_test, y_test)
            train_acc_spec[seed, i] = tr_s
            test_acc_spec[seed, i] = te_s
            gap_spec[seed, i] = tr_s - te_s
            
            # Train Random
            model_rand = ExactSIMClassifier(4, pauli_strings=rand_subset)
            tr_r, te_r = train_and_evaluate(model_rand, X_train, y_train, X_test, y_test)
            train_acc_rand[seed, i] = tr_r
            test_acc_rand[seed, i] = te_r
            gap_rand[seed, i] = tr_r - te_r

    # Stats
    mean_gap_spec = np.mean(gap_spec, axis=0)
    mean_gap_rand = np.mean(gap_rand, axis=0)
    
    mean_tr_spec = np.mean(train_acc_spec, axis=0)
    mean_te_spec = np.mean(test_acc_spec, axis=0)
    
    mean_tr_rand = np.mean(train_acc_rand, axis=0)
    mean_te_rand = np.mean(test_acc_rand, axis=0)
    
    # Plot Gap
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, mean_gap_spec, 'o-', label='Spectral Pruned Gap', color='green')
    plt.plot(k_values, mean_gap_rand, 's-', label='Random Basis Gap', color='red')
    plt.xlabel('Basis Size (k)')
    plt.ylabel('Generalization Gap (Train Acc - Test Acc)')
    plt.title('Overfitting Analysis: Gap widens for Random models as K increases')
    plt.legend()
    plt.xscale('log', base=2)
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/overfitting_gap.png')
    print("Saved plot to results/overfitting_gap.png")
    
    # Detailed Plot (Train vs Test)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, mean_tr_spec, '--', color='green', alpha=0.5, label='Spec Train')
    plt.plot(k_values, mean_te_spec, 'o-', color='green', label='Spec Test')
    plt.plot(k_values, mean_tr_rand, '--', color='red', alpha=0.5, label='Rand Train')
    plt.plot(k_values, mean_te_rand, 's-', color='red', label='Rand Test')
    plt.xlabel('Basis Size (k)')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.legend()
    plt.xscale('log', base=2)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Gap again
    plt.plot(k_values, mean_gap_spec, 'o-', label='Spectral Gap', color='green')
    plt.plot(k_values, mean_gap_rand, 's-', label='Random Gap', color='red')
    plt.xlabel('Basis Size (k)')
    plt.ylabel('Gap')
    plt.title('Gap Analysis')
    plt.legend()
    plt.xscale('log', base=2)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../results/overfitting_detailed.png')
    print("Saved detailed plot to results/overfitting_detailed.png")

if __name__ == "__main__":
    run_overfitting_gap_experiment()
