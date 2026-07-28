import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import urllib.request

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ..utils.pauli_utils import generate_pauli_strings
from ..models.exact_sim_classifier import ExactSIMClassifier
from ..utils.data_loader import load_20newsgroups_projected, download_20newsgroups_manual

from src.utils.seeds import set_seed
set_seed()
# Note: this module previously redefined download_20newsgroups_manual and
# load_20newsgroups_projected here, shadowing the imports above. Both duplicates
# referenced sklearn names this file never imported (fetch_20newsgroups,
# TfidfVectorizer, PCA, StandardScaler) and were behaviourally identical to the
# canonical versions in src/utils/data_loader.py, so they have been removed.

def get_canonical_pattern(s):
    """
    Strips identities and sorts characters.
    e.g. 'IXYZ' -> 'XYZ'
    e.g. 'ZIIZ' -> 'ZZ'
    """
    chars = sorted([c for c in s if c != 'I'])
    if not chars:
        return "I"
    return "".join(chars)

def train_and_analyze(n_qubits, ax):
    print(f"\n--- Analyzing N={n_qubits} ({2**n_qubits} features) ---")
    
    X, y = load_20newsgroups_projected(n_qubits)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Convert to Tensor
    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    
    # Train Exact SIM
    # N=3 to 6
    # N=6 is 64x64 matrices. 4096 terms. Might be slow per epoch.
    # Reduce epochs heavily for N=6? Or just be patient.
    epochs = 50 if n_qubits < 6 else 30 
    
    model = ExactSIMClassifier(n_qubits=n_qubits, n_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Ep {epoch}: Loss={loss.item():.4f}")
        
    print(f"Final Loss: {loss.item():.4f}")
    
    # --- Extract Effectiveness ---
    with torch.no_grad():
        w_vals = model.w.detach() # (K,)
        expectations = model.qnode(model.circuit_weights)
        if isinstance(expectations, (list, tuple)):
            expectations = torch.stack(expectations)
        
        # Effective Weight = w * <P>
        w_eff = w_vals * expectations
        w_eff_abs = torch.abs(w_eff).numpy()
        
    # Cluster by Pattern
    pauli_strings = model.pauli_strings
    
    df = pd.DataFrame({
        'String': pauli_strings,
        'Importance': w_eff_abs
    })
    df['Pattern'] = df['String'].apply(get_canonical_pattern)
    
    # Sum importance
    pattern_sums = df.groupby('Pattern')['Importance'].sum().sort_values(ascending=False)
    
    # Top 8
    top_patterns = pattern_sums.head(8)
    
    # Plot on given ax
    top_patterns.plot(kind='bar', ax=ax, color='teal', alpha=0.7)
    ax.set_title(f"N={n_qubits} (Dim={2**n_qubits})")
    ax.set_ylabel("Effective Importance")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    return pattern_sums

def run_analysis():
    os.makedirs('results', exist_ok=True)
    
    ns = [3, 4, 5, 6]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Iterate
    for i, n in enumerate(ns):
        train_and_analyze(n, axes[i])
        
    plt.tight_layout()
    plt.savefig('results/20newsgroups_canonical_patterns.png')
    print("\nAnalysis complete. Plot saved to results/20newsgroups_canonical_patterns.png")

if __name__ == "__main__":
    run_analysis()
