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

def add_feature_noise(X, noise_level=0.1):
    """
    Adds Gaussian noise to features.
    X: (n_samples, n_features)
    noise_level: std dev of noise relative to data scale (assuming normalized data)
    """
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def train_model(model, X_train, y_train, epochs=100, lr=0.01):
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

def run_feature_noise_experiment(n_seeds=5):
    print("Loading E. Coli (N=4)...")
    X, y = load_ecoli_n4()
    
    # Noise levels to sweep
    noise_levels = np.linspace(0, 1.0, 11) # 0.0 to 1.0
    
    results_full = np.zeros((n_seeds, len(noise_levels)))
    results_spec = np.zeros((n_seeds, len(noise_levels)))
    
    print(f"Running Feature Noise Experiment with {n_seeds} seeds...")
    
    for seed in range(n_seeds):
        print(f"--- Seed {seed+1}/{n_seeds} ---")
        X_train_clean, X_test_clean, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42+seed
        )
        
        # Train on Noisy, Test on Clean (Generalization check)
        for i, noise_lvl in enumerate(noise_levels):
            print(f"  > Noise Level {i+1}/{len(noise_levels)}: {noise_lvl:.2f}", flush=True)
            # Create Noisy Training Data
            X_train_noisy = add_feature_noise(X_train_clean, noise_lvl)
            
            # Generate Spectral basis from the current Noisy Training Data
            if noise_lvl == 0:
                 spec_basis, _, _ = generate_spectral_pauli_strings(X_train_clean, y_train, 4)
            else:
                 spec_basis, _, _ = generate_spectral_pauli_strings(X_train_noisy, y_train, 4)
            
            spec_basis_top32 = spec_basis[:32]
            
            # Full Basis (or random large)
            # For N=4, Full is 256.
            full_basis = generate_pauli_strings(4)
            
            # Train Spectral
            model_spec = ExactSIMClassifier(4, pauli_strings=spec_basis_top32)
            train_model(model_spec, X_train_noisy, y_train)
            
            # Train Full
            model_full = ExactSIMClassifier(4, pauli_strings=full_basis)
            train_model(model_full, X_train_noisy, y_train)
            
            # Evaluate on CLEAN Test Data (True Generalization)
            X_test_t = torch.tensor(X_test_clean, dtype=torch.float64)
            
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
    plt.plot(noise_levels, mean_spec, 'o-', label='Spectral (Top-32)', color='green')
    plt.fill_between(noise_levels, mean_spec - std_spec, mean_spec + std_spec, color='green', alpha=0.15)
    
    plt.plot(noise_levels, mean_full, 's-', label='Full Basis (256)', color='red')
    plt.fill_between(noise_levels, mean_full - std_full, mean_full + std_full, color='red', alpha=0.15)
    
    plt.xlabel('Training Feature Noise Level (std)')
    plt.ylabel('Test Accuracy (on Clean Data)')
    plt.title('Robustness to Feature Noise: Spectral vs Full')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/feature_noise_sweep.png')
    print("Saved plot to results/feature_noise_sweep.png")

if __name__ == "__main__":
    run_feature_noise_experiment()
