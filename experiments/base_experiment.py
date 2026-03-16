import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.utils.pauli_utils import generate_pauli_strings, classify_pauli_string
from src.models.sim_classifier import SIMClassifier
from src.utils.data_generator import generate_dataset

def run_experiment():
    n_qubits = 4 # Keep small for simulation speed (dim=16)
    n_samples = 500
    
    regimes = ['linear', 'pairwise', 'conditional']
    
    # Define Pauli Classes (in order of addition)
    # The prompt asks for: (C0 -> C1Z -> C1X -> C2ZX -> C2XX)
    # Plus "CY" as a check.
    class_order = ['C0', 'C1Z', 'C1X', 'C2ZX', 'C2XX', 'CY']
    
    # Generate all strings first
    all_strings = generate_pauli_strings(n_qubits)
    classified_strings = {s: classify_pauli_string(s) for s in all_strings}
    
    results = []
    
    print(f"Starting Experiment with {n_qubits} qubits...")
    
    for regime in regimes:
        print(f"\n--- Regime: {regime} ---")
        
        # Generate Data
        X_train, y_train = generate_dataset(n_samples, n_qubits, regime, random_state=42, hamiltonian_seed=123)
        X_test, y_test = generate_dataset(200, n_qubits, regime, random_state=99, hamiltonian_seed=123)
        
        current_basis = []
        
        for cls in class_order:
            # 1. Add class to basis
            new_terms = [s for s, c in classified_strings.items() if c == cls]
            if not new_terms:
                print(f"Warning: No terms found for class {cls}")
                continue
                
            current_basis.extend(new_terms)
            print(f"Adding {cls} ({len(new_terms)} terms). Total basis size: {len(current_basis)}")
            
            # 2. Train SIM
            sim = SIMClassifier(pauli_strings=current_basis, C=100.0, random_state=42)
            sim.fit(X_train, y_train)
            
            # 3. Evaluate
            train_acc = sim.score(X_train, y_train)
            test_acc = sim.score(X_test, y_test)
            
            # 4. Analyze Weights
            weights = sim.get_feature_weights()
            # Map weights back to classes
            class_weight_norms = {}
            feature_map = {s: i for i, s in enumerate(current_basis)}
            
            for check_cls in class_order:
                # Find indices for this class in current basis
                indices = [feature_map[s] for s in current_basis if classified_strings[s] == check_cls]
                if indices:
                    norm = np.mean(np.abs(weights[indices])) # Mean absolute weight
                    class_weight_norms[check_cls] = norm
                else:
                    class_weight_norms[check_cls] = 0.0
            
            results.append({
                'Regime': regime,
                'Added_Class': cls,
                'Basis_Size': len(current_basis),
                'Train_Acc': train_acc,
                'Test_Acc': test_acc,
                'Weight_Norm_C0': class_weight_norms.get('C0', 0),
                'Weight_Norm_C1Z': class_weight_norms.get('C1Z', 0),
                'Weight_Norm_C1X': class_weight_norms.get('C1X', 0),
                'Weight_Norm_C2ZX': class_weight_norms.get('C2ZX', 0),
                'Weight_Norm_C2XX': class_weight_norms.get('C2XX', 0),
                'Weight_Norm_CY': class_weight_norms.get('CY', 0)
            })
            
            print(f"  -> Test Acc: {test_acc:.4f} | Top Weights: {[k for k,v in class_weight_norms.items() if v > 0.1]}")

    # Save results
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/experiment_results.csv', index=False)
    print("\nResults saved to results/experiment_results.csv")
    
    # Plotting
    plot_results(df)

def plot_results(df):
    regimes = df['Regime'].unique()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, regime in enumerate(regimes):
        subset = df[df['Regime'] == regime]
        ax = axes[i]
        
        # Plot Accuracy
        ax.plot(subset['Added_Class'], subset['Test_Acc'], marker='o', label='Test Accuracy', linewidth=2)
        ax.set_title(f'Regime: {regime}')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Cumulative Pauli Class')
        ax.grid(True, alpha=0.3)
        
        # Plot Weight Norms (on twin axis?)
        # Or just show what dominates
        pass

    plt.tight_layout()
    plt.savefig('results/accuracy_plot.png')
    print("Plot saved to results/accuracy_plot.png")
    
    # Weight distribution plot
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    for i, regime in enumerate(regimes):
        subset = df[df['Regime'] == regime] # Use the FINAL step (all classes added)
        final_row = subset.iloc[-1]
        
        ax = axes2[i]
        cols = [c for c in df.columns if 'Weight_Norm' in c]
        labels = [c.replace('Weight_Norm_', '') for c in cols]
        vals = [final_row[c] for c in cols]
        
        ax.bar(labels, vals, color='skyblue')
        ax.set_title(f'Final Weight Distribution ({regime})')
        ax.set_ylabel('Mean Abs Weight')
        
    plt.tight_layout()
    plt.savefig('results/weights_plot.png')
    print("Plot saved to results/weights_plot.png")

if __name__ == "__main__":
    run_experiment()
