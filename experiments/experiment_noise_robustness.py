import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from spectral_pauli_generator import generate_spectral_pauli_strings
from analysis_canonical_patterns import load_20newsgroups_projected
from exact_sim_classifier import ExactSIMClassifier

class NoisySIMClassifier(ExactSIMClassifier):
    def __init__(self, n_qubits, n_layers=2, pauli_strings=None, noise_prob=0.0):
        super().__init__(n_qubits, n_layers, pauli_strings=pauli_strings)
        self.noise_prob = noise_prob
        
    def forward(self, x):
        # x: (B, dim)
        x_tilde = x + self.b
        # Compute x^T P x for all P: (B, K)
        # Using registered buffer P_tensor (K, D, D)
        # einsum: b=batch, k=pauli, d=dim1, l=dim2
        # 'bd,kdl,bl->bk'
        # x_tilde[b,d] * P[k,d,l] * x_tilde[b,l] -> Result[b,k]
        quad_features = torch.einsum('bd,kdl,bl->bk', x_tilde, self.P_tensor, x_tilde)
        
        # Get Quantum Expectations <P>_theta
        # These are constant for a fixed theta but "measured" on QPU.
        expectations = self.qnode(self.circuit_weights) # (K,)
        if isinstance(expectations, (list, tuple)):
            expectations = torch.stack(expectations)
            
        # Apply Noise
        if self.noise_prob > 0:
            weights = []
            for s in self.pauli_strings:
                w = sum(1 for c in s if c != 'I')
                weights.append(w)
            weights_t = torch.tensor(weights, device=expectations.device, dtype=expectations.dtype)
            
            # Damping: (1 - p)^w
            damping = (1 - self.noise_prob) ** weights_t
            expectations = expectations * damping
            
        # Combine
        # w_j * (x^T P x) * <P>_noisy
        # (B, K) * (K,) * (K,)
        
        # We need to broadcast expectations
        exp_expanded = expectations.unsqueeze(0) # (1, K)
        w_expanded = self.w.unsqueeze(0) # (1, K)
        
        combined = quad_features * w_expanded * exp_expanded
        
        # Sum over K
        logits = combined.sum(dim=1)
        return torch.sigmoid(logits)

def run_noise_experiment(n_seeds=5):
    print("Loading 20 Newsgroups (N=4)...")
    X, y = load_20newsgroups_projected(n_qubits=4)
    
    noise_levels = np.linspace(0, 0.2, 8) 
    
    # Storage for results: [seed, noise_level]
    results_full = np.zeros((n_seeds, len(noise_levels)))
    results_spec = np.zeros((n_seeds, len(noise_levels)))
    
    avg_weight_full = 0
    avg_weight_spec = 0
    
    print(f"Running Noise Robustness Experiment with {n_seeds} seeds...")
    
    for seed in range(n_seeds):
        print(f"--- Seed {seed+1}/{n_seeds} ---")
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42+seed)
        
        # 1. Spectral Selection
        spectral_strings, _, _ = generate_spectral_pauli_strings(X_train, y_train, 4)
        spectral_50 = spectral_strings[:50]
        full_basis = spectral_strings # All 256
        
        # Weight Analysis (Accumulate)
        w_full = np.mean([sum(1 for c in s if c != 'I') for s in full_basis])
        w_spec = np.mean([sum(1 for c in s if c != 'I') for s in spectral_50])
        avg_weight_full += w_full
        avg_weight_spec += w_spec
        
        # Train Clean Models
        def train_clean(p_strings):
            model = NoisySIMClassifier(4, pauli_strings=p_strings, noise_prob=0.0)
            opt = optim.Adam(model.parameters(), lr=0.01)
            crt = nn.BCELoss()
            X_t = torch.tensor(X_train, dtype=torch.float64)
            y_t = torch.tensor(y_train, dtype=torch.float64)
            for _ in range(60): # Slightly more epochs
                opt.zero_grad()
                loss = crt(model(X_t), y_t)
                loss.backward()
                opt.step()
            return model

        model_full = train_clean(full_basis)
        model_spec = train_clean(spectral_50)
        
        # Test Noise Levels
        X_test_t = torch.tensor(X_test, dtype=torch.float64)
        for i, p in enumerate(noise_levels):
            # Full
            model_full.noise_prob = p
            preds = (model_full(X_test_t) > 0.5).float().detach().numpy()
            results_full[seed, i] = accuracy_score(y_test, preds)
            
            # Spectral
            model_spec.noise_prob = p
            preds = (model_spec(X_test_t) > 0.5).float().detach().numpy()
            results_spec[seed, i] = accuracy_score(y_test, preds)

    # Stats
    avg_weight_full /= n_seeds
    avg_weight_spec /= n_seeds
    
    mean_full = np.mean(results_full, axis=0)
    std_full = np.std(results_full, axis=0)
    
    mean_spec = np.mean(results_spec, axis=0)
    std_spec = np.std(results_spec, axis=0)
    
    print("\n=== Analysis Results ===")
    print(f"Avg Pauli Weight - Full Basis: {avg_weight_full:.2f}")
    print(f"Avg Pauli Weight - Spectral Top-50: {avg_weight_spec:.2f}")
    print("Full Accs (Mean):", np.round(mean_full, 3))
    print("Spec Accs (Mean):", np.round(mean_spec, 3))
    
    # Plot with Error Bars
    plt.figure(figsize=(8, 6))
    
    plt.plot(noise_levels, mean_full, 'o-', label=f'Full Basis (Avg W={avg_weight_full:.1f})', color='blue')
    plt.fill_between(noise_levels, mean_full - std_full, mean_full + std_full, color='blue', alpha=0.15)
    
    plt.plot(noise_levels, mean_spec, 's-', label=f'Spectral Top-50 (Avg W={avg_weight_spec:.1f})', color='orange')
    plt.fill_between(noise_levels, mean_spec - std_spec, mean_spec + std_spec, color='orange', alpha=0.15)
    
    plt.xlabel('Depolarizing Noise Probability (p)')
    plt.ylabel('Test Accuracy')
    plt.title(f'Noise Robustness (Mean of {n_seeds} runs)\nSpectral Selection Favors Lower Weight Terms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/noise_robustness.png', dpi=300)
    print("Saved plot to results/noise_robustness.png")

if __name__ == "__main__":
    run_noise_experiment()
