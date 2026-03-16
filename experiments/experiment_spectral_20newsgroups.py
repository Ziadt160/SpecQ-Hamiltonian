import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.generators.spectral_pauli_generator import generate_spectral_pauli_strings
from src.models.exact_sim_classifier import ExactSIMClassifier
from src.utils.pauli_utils import generate_pauli_strings
from src.experiments.experiment_20newsgroups import load_and_preprocess_20newsgroups_n4

def train_model(X_train, y_train, X_test, y_test, pauli_strings, n_qubits, name):
    print(f"\n--- Training {name} (Terms: {len(pauli_strings)}) ---")
    
    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    X_te = torch.tensor(X_test, dtype=torch.float64)
    
    model = ExactSIMClassifier(n_qubits=n_qubits, n_layers=3, pauli_strings=pauli_strings)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    losses = []
    
    for epoch in range(60): # 60 epochs for comparison
        optimizer.zero_grad()
        out = model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"{name} Epoch {epoch}: Loss={loss.item():.4f}")
            
    # Eval
    model.eval()
    with torch.no_grad():
        preds = (model(X_te).numpy() > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        
    print(f"{name} Final Accuracy: {acc:.4f}")
    return acc, losses

def run_spectral_experiment():
    os.makedirs('results', exist_ok=True)
    
    # 1. Load Data (N=4)
    X, y, n_qubits = load_and_preprocess_20newsgroups_n4()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 2. Generate Spectral Basis (using Training Data ONLY)
    print("\nGenerating Spectral Basis from Training Data...")
    top_k = 50 # Compress 256 -> 50
    spectral_strings, coefs = generate_spectral_pauli_strings(X_train, y_train, n_qubits, top_k=top_k)
    print(f"Top 5 Spectral Strings: {spectral_strings[:5]}")
    
    # 3. Generate Random Basis for Control
    all_strings = generate_pauli_strings(n_qubits)
    import random
    random.seed(42)
    random_strings = random.sample(all_strings, top_k)
    
    # 4. Train Models
    # A. Full Basis (256 terms)
    acc_full, loss_full = train_model(X_train, y_train, X_test, y_test, all_strings, n_qubits, "Full Basis")
    
    # B. Spectral Basis (Top 50)
    acc_spec, loss_spec = train_model(X_train, y_train, X_test, y_test, spectral_strings, n_qubits, f"Spectral Top-{top_k}")
    
    # C. Random Basis (Top 50)
    acc_rand, loss_rand = train_model(X_train, y_train, X_test, y_test, random_strings, n_qubits, f"Random Top-{top_k}")
    
    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(loss_full, label=f'Full (256) Acc={acc_full:.3f}', color='black', alpha=0.7)
    plt.plot(loss_spec, label=f'Spectral (50) Acc={acc_spec:.3f}', color='blue', linewidth=2)
    plt.plot(loss_rand, label=f'Random (50) Acc={acc_rand:.3f}', color='red', linestyle='dashed')
    
    plt.title(f'Spectral Basis Generation (N=4, Top-{top_k}) on 20 Newsgroups')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/spectral_experiment_convergence.png')
    print("Plot saved to results/spectral_experiment_convergence.png")
    
    # Save text
    with open('results/spectral_experiment_results.txt', 'w') as f:
        f.write(f"Full Basis Accuracy: {acc_full:.4f}\n")
        f.write(f"Spectral Top-{top_k} Accuracy: {acc_spec:.4f}\n")
        f.write(f"Random Top-{top_k} Accuracy: {acc_rand:.4f}\n")

if __name__ == "__main__":
    run_spectral_experiment()
