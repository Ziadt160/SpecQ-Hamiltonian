import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.experiment_ecoli_reduced import load_and_reduce_ecoli
from src.utils.data_loader import load_ecoli_reduced as load_ecoli_n4
from src.utils.pauli_utils import generate_pauli_strings, classify_pauli_string
from src.models.sim_classifier import SIMClassifier
from src.models.exact_sim_classifier import ExactSIMClassifier

def train_exact_sim(X_train, y_train, X_test, y_test, epochs=300):
    print("\n--- Training Exact SIM (PyTorch + PennyLane) ---")
    
    # Convert to Tensor (Double precision for PennyLane compatibility)
    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    X_te = torch.tensor(X_test, dtype=torch.float64)
    y_te = torch.tensor(y_test, dtype=torch.float64)
    
    model = ExactSIMClassifier(n_qubits=4, n_layers=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tr) # (B,)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
    # Eval
    with torch.no_grad():
        preds = model(X_te).numpy()
        preds_cls = (preds > 0.5).astype(int)
        acc = accuracy_score(y_test, preds_cls)
        
    print(f"Exact SIM Final Accuracy: {acc:.4f}")
    return acc, losses

def run_comparison():
    X, y = load_ecoli_n4()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. Classical SIM (Baseline)
    print("\n--- Training Classical SIM (Logistic Regression) ---")
    # Generate strings for N=4
    strs = generate_pauli_strings(4)
    sim_classic = SIMClassifier(pauli_strings=strs, C=10.0, random_state=42)
    sim_classic.fit(X_train, y_train)
    acc_classic = sim_classic.score(X_test, y_test)
    print(f"Classical SIM Accuracy: {acc_classic:.4f}")
    
    # 2. Exact SIM
    acc_exact, losses = train_exact_sim(X_train, y_train, X_test, y_test)
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Exact SIM Loss')
    plt.axhline(y=0, color='grey', linestyle='--', alpha=0.3)
    plt.title(f"Exact SIM Training (acc={acc_exact:.3f}) vs Classical (acc={acc_classic:.3f})")
    plt.xlabel('Epoch')
    plt.ylabel('BCELoss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/exact_sim_convergence.png')
    print("Plot saved to results/exact_sim_convergence.png")
    
    # Save Results
    with open('results/exact_comparison.txt', 'w') as f:
        f.write(f"Classical SIM (N=4): {acc_classic:.4f}\n")
        f.write(f"Exact SIM (N=4): {acc_exact:.4f}\n")

if __name__ == "__main__":
    run_comparison()
