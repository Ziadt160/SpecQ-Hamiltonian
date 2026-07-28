import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.spectral_sim_classifier import SpectralExactSIMClassifier
from src.models.exact_sim_classifier import ExactSIMClassifier
from src.utils.data_utils import load_fraud_data
from src.models.vqc_classifier import NoisyVQC
from src.models.noisy_qsvm import NoisyQSVM

from src.utils.seeds import set_seed
set_seed()

def run_noise_study():
    print("Starting Noise Resilience Study (Quantum)...")
    os.makedirs('results/case_study_fraud', exist_ok=True)
    
    # Use unified loader
    X_norm, y = load_fraud_data(n_samples=500, n_qubits=4)
    
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
    X_te_t = torch.tensor(X_test, dtype=torch.float64)
    
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
    csv_path = 'results/case_study_fraud/noise_results_comprehensive.csv'
    
    # Initialize results container and load existing results to RESUME
    results = {k: [] for k in ['p', 'Model', 'Train_AUC', 'Train_F1', 'Test_AUC', 'Test_F1', 'Train_Time']}
    existing_results = set()
    
    if os.path.exists(csv_path):
        try:
            df_curr = pd.read_csv(csv_path)
            for _, row in df_curr.iterrows():
                existing_results.add((round(float(row['p']), 4), row['Model']))
            results = df_curr.to_dict('list')
            print(f"Loaded {len(existing_results)} existing model-level results from {csv_path}.")
        except Exception as e:
            print(f"Could not load existing CSV ({e}), starting fresh.")
    else:
        # Initialize CSV with headers if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame(columns=results.keys()).to_csv(csv_path, index=False)

    # Models that require pre-processing (clean training) once
    base_models = {} # {label: model_object}
    base_times = {}

    for p in noise_levels:
        p_round = round(p, 4)
        
        # Check which models at this noise level are already completed
        models_to_run = [
            'Spectral SIM (Flipped)',
            'VQC (StronglyEntangling)',
            'VQC (BasicEntangling)',
            'QSVM (AmplitudeEmbedding)'
        ]
        
        pending_models = [m for m in models_to_run if (p_round, m) not in existing_results]
        
        if not pending_models:
            print(f"Skipping noise level p={p_round} (All models complete).")
            continue
            
        print(f"Processing Noise Level p={p_round}...")

        # 1. Ensure Base Models (Clean) are trained if needed for weights
        # We need clean weights for SIM and VQC. QSVM fits per level.
        if not base_models:
            print("Training base clean models for weights...")
            
            # SIM Base
            start_t = time.time()
            sim_base = SpectralExactSIMClassifier(n_qubits=4, n_layers=2, top_k=20)
            sim_base.fit(X_train, y_train, epochs=20, verbose=False)
            base_models['Spectral SIM (Flipped)'] = sim_base
            base_times['Spectral SIM (Flipped)'] = time.time() - start_t
            
            # VQC Bases
            X_tr_t = torch.tensor(X_train, dtype=torch.float64)
            y_tr_t = torch.tensor(y_train, dtype=torch.float64)
            
            for variant, label in [('StronglyEntangling', 'VQC (StronglyEntangling)'), 
                                   ('BasicEntangling', 'VQC (BasicEntangling)')]:
                start_t = time.time()
                vqc_b = NoisyVQC(n_qubits=4, variant=variant, noise_prob=0.0)
                opt = torch.optim.Adam(vqc_b.parameters(), lr=0.01)
                for _ in range(20):
                    opt.zero_grad()
                    loss = nn.BCELoss()(vqc_b(X_tr_t), y_tr_t)
                    loss.backward()
                    opt.step()
                base_models[label] = vqc_b
                base_times[label] = time.time() - start_t
            
            print("Base clean models trained.")

        # 2. Evaluate each pending model
        for label in pending_models:
            print(f"  Evaluating {label} at p={p_round}...")
            
            t_train = 0
            if label == 'QSVM (AmplitudeEmbedding)':
                model = NoisyQSVM(n_qubits=4, noise_prob=p)
                start_t = time.time()
                model.fit(X_train[:100], y_train[:100])
                t_train = time.time() - start_t
                
                probs_test = model.predict_proba(X_test)[:, 1]
                probs_train = model.predict_proba(X_train[:100])[:, 1]
                y_tr_sub = y_train[:100]
            else:
                # Spectral SIM or VQC - load weights from base
                if 'Spectral' in label:
                    model = SpectralExactSIMClassifier(n_qubits=4, n_layers=2, top_k=20, 
                                                        noise_prob=p, 
                                                        pauli_strings=base_models[label].pauli_strings_)
                    # Initialize inner model structure
                    model.fit(X_train[:2], y_train[:2], epochs=0, verbose=False)
                    model.model.load_state_dict(base_models[label].model.state_dict())
                else:
                    variant = 'StronglyEntangling' if 'Strong' in label else 'BasicEntangling'
                    model = NoisyVQC(n_qubits=4, variant=variant, noise_prob=p)
                    model.load_state_dict(base_models[label].state_dict())
                
                t_train = base_times[label]
                with torch.no_grad():
                    probs_test = model(X_te_t).numpy()
                    probs_train = model(X_tr_t).numpy()
                y_tr_sub = y_train

            # Metrics
            preds_test = (probs_test > 0.5).astype(int)
            preds_train = (probs_train > 0.5).astype(int)
            
            auc_test = roc_auc_score(y_test, probs_test)
            f1_test = f1_score(y_test, preds_test)
            auc_train = roc_auc_score(y_tr_sub, probs_train)
            f1_train = f1_score(y_tr_sub, preds_train)
            
            # Update results and save incrementally
            results['p'].append(p)
            results['Model'].append(label)
            results['Train_AUC'].append(auc_train)
            results['Train_F1'].append(f1_train)
            results['Test_AUC'].append(auc_test)
            results['Test_F1'].append(f1_test)
            results['Train_Time'].append(t_train)
            
            print(f"    -> Train AUC: {auc_train:.4f}, Test AUC: {auc_test:.4f}")
            pd.DataFrame(results).to_csv(csv_path, index=False)
            
        print(f"Noise level p={p_round} sweep complete.")
    # Visualization
    df_results = pd.DataFrame(results)
    
    models = df_results['Model'].unique()
    
    # 1. AUC Comparison (Train vs Test)
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 5*len(models)))
    if len(models) == 1: axes = [axes]
    
    for i, label in enumerate(models):
        subset = df_results[df_results['Model'] == label]
        axes[i].plot(subset['p'], subset['Train_AUC'], 'o-', label='Train AUC')
        axes[i].plot(subset['p'], subset['Test_AUC'], 's--', label='Test AUC')
        axes[i].set_title(f'Noise Impact: {label}')
        axes[i].set_xlabel('Noise p')
        axes[i].set_ylabel('AUC')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/case_study_fraud/noise_overfitting_analysis.png')
    
    # 2. Global Test AUC comparison
    plt.figure(figsize=(10, 6))
    for label in models:
        subset = df_results[df_results['Model'] == label]
        plt.plot(subset['p'], subset['Test_AUC'], 'o-', label=label, linewidth=2)
    
    plt.xlabel('Quantum Noise Probability (p)')
    plt.ylabel('Test AUC')
    plt.title('Comprehensive Noise Resilience comparison (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/case_study_fraud/noise_test_auc_comparison.png')
    
    print("Noise study analysis completed and plots saved.")

if __name__ == '__main__':
    run_noise_study()
