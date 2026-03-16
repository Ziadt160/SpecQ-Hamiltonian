import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine, load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.exact_sim_classifier import ExactSIMClassifier
from src.experiment_ecoli_exact_ablation import load_ecoli_n4

# --- Constants ---
TOP_32_STRINGS = [
    "IXZZ", "IXXX", "XIZZ", "ZYYZ", "IIII", "YYXZ", "ZXIZ", "IZZX", 
    "ZZZX", "ZZXZ", "YYXI", "XIZI", "IYIY", "XXIZ", "ZYXY", "YZYZ", 
    "YXZY", "XYXY", "IYYX", "ZZXI", "IXZI", "IZXZ", "ZXXX", "XZXI", 
    "XZII", "YIZY", "YXYZ", "IZXX", "YXYX", "ZIII", "IXXI", "IIYY"
]
# Ensure we have exactly 32
assert len(TOP_32_STRINGS) == 32, f"Expected 32 strings, got {len(TOP_32_STRINGS)}"

# --- Data Loaders ---
def load_wine_n4():
    print("Loading Wine dataset...")
    data = load_wine()
    X = data.data
    y = data.target
    
    # Wine has 13 features. We need 16.
    # We can pad with zeros or noise? Or use PCA/Selection to map to 16?
    # Simple strategy: Pad with 3 zeros.
    # Or better: Pad first, then normalize.
    
    # Actually, let's just pad with zeros to reach 16
    n_samples, n_features = X.shape
    X_padded = np.zeros((n_samples, 16))
    X_padded[:, :n_features] = X
    
    # Normalize L2
    os_norms = np.linalg.norm(X_padded, axis=1, keepdims=True)
    os_norms[os_norms == 0] = 1.0
    X_norm = X_padded / os_norms
    
    # Binary classification for simplicity? Or Multi-class?
    # ExactSIM is binary (Sigmoid). Let's do Class 0 vs Rest for now.
    y = (y == 0).astype(int) 
    
    return X_norm, y

def load_mnist_n4():
    print("Loading MNIST (Digits) dataset...")
    data = load_digits()
    X = data.data # (1797, 64)
    y = data.target
    
    # 8x8 images = 64 features.
    # PCA to 16
    pca = PCA(n_components=16)
    X_reduced = pca.fit_transform(X)
    
    # Normalize L2
    norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X_reduced / norms
    
    # Binary: Digit 0 vs Rest
    y = (y == 0).astype(int)
    
    return X_norm, y

# --- Training / Eval Loop ---
def evaluate_dataset(name, load_fn):
    print(f"\n=== Evaluating on {name} ===")
    X, y = load_fn()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. Train with Top 32 E. coli Strings
    print(f"Training on {name} with Top 32 E. coli Pauli Strings...")
    model = ExactSIMClassifier(n_qubits=4, n_layers=3, pauli_strings=TOP_32_STRINGS)
    
    # Convert to Tensor
    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    X_te = torch.tensor(X_test, dtype=torch.float64)
    y_te = torch.tensor(y_test, dtype=torch.float64)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for epoch in range(200): # 200 epochs should be enough
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()
        
    # Eval
    with torch.no_grad():
        preds = model(X_te).numpy()
        preds_cls = (preds > 0.5).astype(int)
        acc = accuracy_score(y_test, preds_cls)
        
    print(f"{name} Results -> Accuracy: {acc:.4f}")
    return acc

def run_universal_experiment():
    results = {}
    
    # 1. E. coli (Baseline check)
    acc_ecoli = evaluate_dataset("E. Coli (CTZ)", load_ecoli_n4)
    results['EColi'] = acc_ecoli

    # 2. Wine
    acc_wine = evaluate_dataset("Wine (Class 0)", load_wine_n4)
    results['Wine'] = acc_wine
    
    # 3. MNIST
    acc_mnist = evaluate_dataset("MNIST (Digit 0)", load_mnist_n4)
    results['MNIST'] = acc_mnist
    
    # Save Report
    print("\n--- Universal Experiment Summary ---")
    print(f"Pauli Set: Top 32 from E. coli Experiment")
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")
        
    with open('results/universal_pauli_results.txt', 'w') as f:
        f.write("Universal Pauli Set Experiment (Top 32 E. coli Strings)\n")
        f.write("====================================================\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc:.4f}\n")

if __name__ == "__main__":
    run_universal_experiment()
