import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.spectral_sim_classifier import SpectralExactSIMClassifier
from src.utils.data_utils import load_fraud_data

from src.utils.seeds import set_seed
set_seed()

def run_topk_noise_sweep():
    print("Starting Top-K Noise Resilience Sweep (Spectral SIM)...")
    os.makedirs('results/case_study_fraud', exist_ok=True)
    
    # Use unified loader
    X_norm, y = load_fraud_data(n_samples=500, n_qubits=4)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
    X_te_t = torch.tensor(X_test, dtype=torch.float64)
    X_tr_t = torch.tensor(X_train, dtype=torch.float64)

    noise_levels = [0.0, 0.05, 0.1, 0.2]
    # top_k sequence based on user input 4 8 6 12 14 18...
    top_k_values = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
    
    csv_path = 'results/case_study_fraud/topk_noise_results.csv'
    
    # Initialize results container 
    results = {k: [] for k in ['top_k', 'p', 'Train_AUC', 'Train_F1', 'Test_AUC', 'Test_F1', 'Train_Time']}
    existing_results = set()
    
    if os.path.exists(csv_path):
        try:
            df_curr = pd.read_csv(csv_path)
            for _, row in df_curr.iterrows():
                existing_results.add((int(row['top_k']), round(float(row['p']), 4)))
            results = df_curr.to_dict('list')
            print(f"Loaded {len(existing_results)} existing (top_k, p) results from {csv_path}.")
        except Exception as e:
            print(f"Could not load existing CSV ({e}), starting fresh.")
    else:
        pd.DataFrame(columns=results.keys()).to_csv(csv_path, index=False)

    for k in top_k_values:
        print(f"\nEvaluating Top-K = {k}...")
        
        # We need a clean model for EACH k to get correct basis/weights
        clean_model = None
        clean_time = 0
        
        for p in noise_levels:
            p_round = round(p, 4)
            
            if (k, p_round) in existing_results:
                print(f"  Skipping (k={k}, p={p_round}) - already in CSV.")
                continue
                
            # 1. Ensure Clean Base Model for this k is trained
            if clean_model is None:
                print(f"  Training clean base model (k={k})...")
                start_t = time.time()
                clean_model = SpectralExactSIMClassifier(n_qubits=4, n_layers=2, top_k=k)
                clean_model.fit(X_train, y_train, epochs=20, verbose=False)
                clean_time = time.time() - start_t
            
            # 2. Initialize Noisy Model with same basis/weights
            print(f"  Sweeping noise p={p_round}...")
            model = SpectralExactSIMClassifier(
                n_qubits=4, n_layers=2, top_k=k, 
                noise_prob=p, 
                pauli_strings=clean_model.pauli_strings_
            )
            # Initialize inner model structure
            model.fit(X_train[:2], y_train[:2], epochs=0, verbose=False)
            model.model.load_state_dict(clean_model.model.state_dict())
            
            # 3. Evaluate
            with torch.no_grad():
                probs_test = model(X_te_t).numpy()
                probs_train = model(X_tr_t).numpy()
            
            preds_test = (probs_test > 0.5).astype(int)
            preds_train = (probs_train > 0.5).astype(int)
            
            auc_test = roc_auc_score(y_test, probs_test)
            f1_test = f1_score(y_test, preds_test)
            auc_train = roc_auc_score(y_train, probs_train)
            f1_train = f1_score(y_train, preds_train)
            
            # Update results
            results['top_k'].append(k)
            results['p'].append(p)
            results['Train_AUC'].append(auc_train)
            results['Train_F1'].append(f1_train)
            results['Test_AUC'].append(auc_test)
            results['Test_F1'].append(f1_test)
            results['Train_Time'].append(clean_time)
            
            print(f"    (k={k}, p={p}) -> Test AUC: {auc_test:.4f}")
            pd.DataFrame(results).to_csv(csv_path, index=False)

    # Visualization
    print("\nGenerating Plots...")
    df_results = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    for p in sorted(df_results['p'].unique()):
        subset = df_results[df_results['p'] == p].sort_values('top_k')
        plt.plot(subset['top_k'], subset['Test_AUC'], 'o-', label=f'Noise p={p}', linewidth=2)
    
    plt.xlabel('Number of Top-K Pauli Strings')
    plt.ylabel('Test AUC')
    plt.title('Impact of Model Complexity (Top-K) on Noise Resilience')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/case_study_fraud/topk_noise_resilience.png')
    
    plt.figure(figsize=(10, 6))
    for p in sorted(df_results['p'].unique()):
        subset = df_results[df_results['p'] == p].sort_values('top_k')
        plt.plot(subset['top_k'], subset['Test_F1'], 's--', label=f'Noise p={p}', linewidth=2)
    
    plt.xlabel('Number of Top-K Pauli Strings')
    plt.ylabel('Test F1 Score')
    plt.title('Impact of Model Complexity (Top-K) on Noise Resilience (F1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/case_study_fraud/topk_noise_resilience_f1.png')
    
    print("Sweep complete. Results and plots saved to results/case_study_fraud/.")

if __name__ == '__main__':
    run_topk_noise_sweep()
