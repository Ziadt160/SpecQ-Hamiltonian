import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.utils.data_loader import load_20newsgroups_projected
from src.generators.spectral_pauli_generator import get_adaptive_spectral_paulis
# train_model_k lives in stress_test_spectral; this module previously imported a
# nonexistent 'load_ecoli_n4_model_k' and then called train_model_k undefined.
from src.analysis.stress_test_spectral import train_model_k


from src.utils.seeds import set_seed
set_seed()

def run_adaptive_experiment():
    os.makedirs('results', exist_ok=True)
    n_qubits = 6 # N=6
    print(f"Loading Data N={n_qubits}...")
    X, y = load_20newsgroups_projected(n_qubits)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    etas = [0.95, 0.98]
    
    print("\n--- Adaptive Spectral Cutoff Experiment ---")
    
    results = []
    
    for eta in etas:
        print(f"\nProcessing eta={eta}...")
        strings, coefs, k = get_adaptive_spectral_paulis(X_train, y_train, n_qubits, eta=eta)
        print(f"Selected k={k} terms (out of 4096) for eta={eta} (Compression: {100*(1 - k/4096):.1f}%)")
        
        acc = train_model_k(X_train, y_train, X_test, y_test, strings, n_qubits, k)
        results.append((eta, k, acc))
        
    print("\n--- Summary ---")
    for r in results:
        print(f"Eta={r[0]}: k={r[1]}, Acc={r[2]:.4f}")
        
    # Save to text
    with open('results/spectral_adaptive_results.txt', 'w') as f:
        f.write("Adaptive Spectral Cutoff (N=6)\n")
        for r in results:
            f.write(f"Eta={r[0]}: k={r[1]}, Acc={r[2]:.4f}\n")

if __name__ == "__main__":
    run_adaptive_experiment()
