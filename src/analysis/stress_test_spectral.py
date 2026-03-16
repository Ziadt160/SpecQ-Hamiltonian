import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

from ..utils.data_loader import load_20newsgroups_projected
from ..models.exact_sim_classifier import ExactSIMClassifier
from ..generators.spectral_pauli_generator import generate_spectral_pauli_strings

def train_model_k(X_train, y_train, X_test, y_test, pauli_strings, n_qubits, k):
    print(f"\n--- Training Top-{k} (Terms: {len(pauli_strings)}) ---")
    
    # Check dimensions
    if len(pauli_strings) == 0:
        return 0.5 # Random guessing
        
    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    X_te = torch.tensor(X_test, dtype=torch.float64)
    
    model = ExactSIMClassifier(n_qubits=n_qubits, n_layers=3, pauli_strings=pauli_strings)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    # 40 epochs for speed in sweep
    for epoch in range(40):
        optimizer.zero_grad()
        out = model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()
            
    # Eval
    model.eval()
    with torch.no_grad():
        preds = (model(X_te).numpy() > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        
    print(f"Top-{k} Accuracy: {acc:.4f}")
    return acc

def run_stress_test():
    os.makedirs('results', exist_ok=True)
    
    n_qubits = 6 # 64 dims, 4096 potential terms
    print(f"Loading Data for N={n_qubits}...")
    X, y = load_20newsgroups_projected(n_qubits)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Generating Full Spectral Basis Ranking...")
    # Get enough to cover our max k
    max_k = 256
    # Note: Generating ALL 4096 spectral scores might take a moment.
    # Let's generate all to be safe? 4096 is fine.
    # Actually generate_spectral_pauli_strings defaults to return all if top_k None
    # But let's ask for max_k + buffer
    spectral_strings, coefs = generate_spectral_pauli_strings(X_train, y_train, n_qubits, top_k=max_k)
    
    k_values = [2, 4, 8, 16, 32, 64, 128, 256]
    accuracies = []
    
    for k in k_values:
        start_k = 0
        current_strings = spectral_strings[start_k:k]
        
        acc = train_model_k(X_train, y_train, X_test, y_test, current_strings, n_qubits, k)
        accuracies.append(acc)
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, color='crimson')
    plt.xscale('log', base=2)
    plt.xlabel('Number of Pauli Terms (k)')
    plt.ylabel('Test Accuracy')
    plt.title(f'Spectral Algorithm Stress Test (N={n_qubits})')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xticks(k_values, [str(k) for k in k_values])
    
    plt.savefig('results/spectral_stress_test.png')
    print("Plot saved to results/spectral_stress_test.png")
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame({'k': k_values, 'accuracy': accuracies})
    df.to_csv('results/spectral_stress_test.csv', index=False)

if __name__ == "__main__":
    run_stress_test()
