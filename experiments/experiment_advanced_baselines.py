
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pauli_utils import generate_pauli_strings
from spectral_pauli_generator import generate_spectral_pauli_strings
from analysis_canonical_patterns import load_20newsgroups_projected
from exact_sim_classifier import ExactSIMClassifier

# Mock the NoisySIMClassifier behavior but without noise for clean baseline comparison
class BaselineSIMClassifier(ExactSIMClassifier):
    def __init__(self, n_qubits, pauli_strings):
        super().__init__(n_qubits, pauli_strings=pauli_strings) # Inherit from ExactSIM
    
    # Use ExactSIM forward

def get_locality_basis(n_qubits, k=50):
    """
    Selects top-k Pauli strings based on lowest weight (Locality heuristic).
    Tie-breaking is random.
    """
    all_strings = generate_pauli_strings(n_qubits)
    # Calculate weights
    weighted_strings = []
    for s in all_strings:
        w = sum(1 for c in s if c != 'I')
        weighted_strings.append((w, s))
    
    # Sort by weight ascending
    weighted_strings.sort(key=lambda x: x[0])
    
    # Take top k strings
    return [s for w, s in weighted_strings[:k]]

def train_and_eval(name, strings, X_train, X_test, y_train, y_test, n_qubits=4):
    print(f"\nTraining {name} (k={len(strings)})...")
    model = BaselineSIMClassifier(n_qubits, strings)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    X_t = torch.tensor(X_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    X_test_t = torch.tensor(X_test, dtype=torch.float64)
    
    for epoch in range(60):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        
    # Eval
    probs = model(X_test_t).detach().numpy()
    preds = (probs > 0.5).astype(float)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0.5
        
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC:      {auc:.4f}")
    return acc, f1, auc

def run_advanced_baselines():
    print("Loading Data (N=4)...")
    X, y = load_20newsgroups_projected(n_qubits=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    k = 50
    
    # 1. Spectral Basis
    print("Generating Spectral Basis...")
    spectral_strings, _, _ = generate_spectral_pauli_strings(X_train, y_train, 4)
    spectral_basis = spectral_strings[:k]
    
    # 2. Locality (Low-Weight) Basis
    print("Generating Locality (Low-Weight) Basis...")
    locality_basis = get_locality_basis(4, k=k)
    
    # 3. Random Basis
    print("Generating Random Basis...")
    all_strings = generate_pauli_strings(4)
    np.random.shuffle(all_strings)
    random_basis = all_strings[:k]
    
    # Comparisons
    results = {}
    results['Spectral'] = train_and_eval("Spectral SIM", spectral_basis, X_train, X_test, y_train, y_test)
    results['Locality'] = train_and_eval("Locality SIM", locality_basis, X_train, X_test, y_train, y_test)
    results['Random']   = train_and_eval("Random SIM", random_basis, X_train, X_test, y_train, y_test)
    
    print("\n=== Final Benchmark Results ===")
    print(f"{'Model':<15} {'Acc':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 45)
    for name, (acc, f1, auc) in results.items():
        print(f"{name:<15} {acc:<10.4f} {f1:<10.4f} {auc:<10.4f}")

if __name__ == "__main__":
    run_advanced_baselines()
