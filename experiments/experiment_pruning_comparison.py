import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_selection import SelectKBest, chi2

from experiment_20newsgroups import load_and_preprocess_20newsgroups_n4
from src.models.exact_sim_classifier import ExactSIMClassifier
from src.generators.spectral_pauli_generator import generate_spectral_pauli_strings
from src.generators.qmi_pauli_generator import generate_qmi_pauli_strings
from src.utils.pauli_utils import generate_pauli_strings

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ecoli_data(n_qubits=4):
    """
    Loads E. Coli data and reduces to 2^n_qubits dimensions.
    """
    print(f"Loading E. Coli dataset (N={n_qubits})...")
    # Path might vary, using the one seen in experiment_ecoli_exact.py
    path = r'd:\Evoth Labs\SIM-Flipped Models\data\EColi_Merged_df.csv'
    if not os.path.exists(path):
        # Falback relative path
        path = 'data/EColi_Merged_df.csv'
        
    df = pd.read_csv(path)
    df = df.dropna(subset=['CTZ'])
    y = df['CTZ'].apply(lambda x: 1 if x == 'R' else 0).values
    
    # Feature Selection
    cols = list(df.columns)
    try: start_idx = cols.index('CIP') + 1
    except: start_idx = 15
    X_genes = df.iloc[:, start_idx:].values
    
    target_dim = 2**n_qubits
    selector = SelectKBest(chi2, k=target_dim)
    X_reduced = selector.fit_transform(X_genes, y)
    
    # L2 Norm per sample
    norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X_reduced / norms
    
    return X_norm, y, n_qubits

def train_and_eval(X_train, y_train, X_test, y_test, pauli_strings, n_qubits, epochs=50):
    # Convert to Tensor (Double precision)
    X_tr = torch.tensor(X_train, dtype=torch.float64).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float64).to(device)
    X_te = torch.tensor(X_test, dtype=torch.float64).to(device)
    
    model = ExactSIMClassifier(n_qubits=n_qubits, n_layers=3, pauli_strings=pauli_strings).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()
        
    # Eval
    model.eval()
    with torch.no_grad():
        preds_prob = model(X_te).cpu().numpy()
        preds_cls = (preds_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_test, preds_cls)
        f1 = f1_score(y_test, preds_cls)
        try:
            auc = roc_auc_score(y_test, preds_prob)
        except:
            auc = 0.5
            
    return acc, f1, auc

def run_ablation_study():
    os.makedirs('results', exist_ok=True)
    results = []
    
    # Configuration
    datasets = [
        ('20Newsgroups', 4, load_and_preprocess_20newsgroups_n4),
        ('EColi', 4, lambda: load_ecoli_data(n_qubits=4)),
        # Uncomment for N=6 if environment supports large matmuls (might be slow)
        # ('EColi', 6, lambda: load_ecoli_data(n_qubits=6)) 
    ]
    
    k_values = [32, 50, 128] # Pruning levels
    
    for ds_name, n_qubits, load_fn in datasets:
        print(f"\n=== Dataset: {ds_name} (N={n_qubits}) ===")
        X, y, _ = load_fn() if ds_name == '20Newsgroups' else (*load_fn(),)
        if len(y) != len(X): # Handle return differences
             X, y = load_fn()
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 1. Generate Rankings
        print("Generating Rankings...")
        
        # Spectral
        spec_strs, _, _ = generate_spectral_pauli_strings(X_train, y_train, n_qubits)
        
        # QMI
        qmi_strs, _ = generate_qmi_pauli_strings(X_train, y_train, n_qubits)
        
        # Random
        all_strs = generate_pauli_strings(n_qubits)
        rand_strs = list(all_strs)
        random.shuffle(rand_strs)
        
        # Methods map
        methods = {
            'Spectral': spec_strs,
            'QMI': qmi_strs,
            'Random': rand_strs
        }
        
        for k in k_values:
            if k > len(all_strs):
                continue
                
            print(f"\n--- k={k} ---")
            for method_name, ranked_strs in methods.items():
                selected_paulis = ranked_strs[:k]
                
                # Train
                print(f"Training {method_name}...")
                acc, f1, auc = train_and_eval(X_train, y_train, X_test, y_test, selected_paulis, n_qubits)
                
                print(f"  Result: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                
                results.append({
                    'dataset': ds_name,
                    'n_qubits': n_qubits,
                    'k': k,
                    'method': method_name,
                    'accuracy': acc,
                    'f1': f1,
                    'auc': auc
                })
                
    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv('results/pruning_ablation_results.csv', index=False)
    print("\nResults saved to results/pruning_ablation_results.csv")
    
    # Plotting Accuracy vs k
    for ds_name in df_res['dataset'].unique():
        subset = df_res[df_res['dataset'] == ds_name]
        plt.figure(figsize=(8, 6))
        for method in subset['method'].unique():
            data = subset[subset['method'] == method]
            data = data.sort_values('k')
            plt.plot(data['k'], data['accuracy'], marker='o', label=method)
            
        plt.title(f'Pruning Method Comparison ({ds_name})')
        plt.xlabel('Number of Paulis (k)')
        plt.ylabel('Test Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'results/pruning_comparison_{ds_name}.png')
        print(f"Plot saved to results/pruning_comparison_{ds_name}.png")

if __name__ == "__main__":
    run_ablation_study()
