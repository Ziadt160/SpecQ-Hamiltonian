import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from src.utils.pauli_utils import generate_pauli_strings, classify_pauli_string
from src.models.sim_classifier import SIMClassifier
from src.experiment_wine import load_and_preprocess_wine

def categorize_string_structure(s):
    """
    Returns a descriptive structure of the string.
    e.g., 'IZXI' -> Weight 2, Type 'ZX'
    """
    weight = sum(1 for c in s if c != 'I')
    if weight == 0:
        return 0, "Identity"
    
    # Extract only non-I chars sorted to make type generic (e.g. ZI vs IZ -> both Z)
    chars = sorted([c for c in s if c != 'I'])
    s_type = "".join(chars)
    
    return weight, s_type

def run_full_experiment():
    X, y, n_qubits = load_and_preprocess_wine()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. Generate ALL strings
    all_strings = generate_pauli_strings(n_qubits)
    print(f"Total Pauli Strings (N={n_qubits}): {len(all_strings)}")
    
    # 2. Train Full Model (Feature selection approach)
    print("\n--- Training Full Model (All Features) ---")
    sim_full = SIMClassifier(pauli_strings=all_strings, C=10.0, random_state=42)
    sim_full.fit(X_train, y_train)
    full_acc = sim_full.score(X_test, y_test)
    print(f"Full Basis Test Accuracy: {full_acc:.4f}")
    
    # Analyze Weights
    # Assuming multiclass (One vs Rest), weights are (n_classes, n_features)
    # We take mean(abs(weight)) across classes to see overall importance
    weights = sim_full.classifier.coef_ # shape (3, 256)
    mean_weights = np.mean(np.abs(weights), axis=0) # shape (256,)
    
    # Create DataFrame of Features
    feat_data = []
    for i, s in enumerate(all_strings):
        w_val = mean_weights[i]
        weight_order, s_type = categorize_string_structure(s)
        feat_data.append({
            'String': s,
            'Weight': weight_order,
            'Type': s_type,
            'Importance': w_val
        })
        
    df_feats = pd.DataFrame(feat_data).sort_values('Importance', ascending=False)
    
    print("\n--- Top 15 Most Important Pauli Terms ---")
    print(df_feats.head(15))
    
    df_feats.to_csv('results/wine_full_feature_importance.csv', index=False)
    
    # 3. Ablation by Order (Weight)
    print("\n--- Ablation by Interaction Order ---")
    results = []
    max_weight = n_qubits
    
    current_basis = []
    for w in range(max_weight + 1):
        # Add all terms of weight w
        new_terms = [s for s in all_strings if sum(1 for c in s if c != 'I') == w]
        current_basis.extend(new_terms)
        
        sim = SIMClassifier(pauli_strings=current_basis, C=10.0, random_state=42)
        sim.fit(X_train, y_train)
        acc = sim.score(X_test, y_test)
        
        print(f"Order {w} (Added {len(new_terms)} terms -> Total {len(current_basis)}): Test Acc = {acc:.4f}")
        results.append({
            'Order': w,
            'Basis_Size': len(current_basis),
            'Test_Acc': acc
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv('results/wine_full_order_ablation.csv', index=False)
    
    plot_full_results(df_feats, df_res)

def plot_full_results(df_feats, df_res):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Ablation by Order
    ax = axes[0]
    ax.plot(df_res['Order'], df_res['Test_Acc'], marker='o', linewidth=2, color='purple')
    ax.set_title('Accuracy vs Interaction Order')
    ax.set_xlabel('Pauli Weight (Interaction Order)')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Top Features Type Distribution
    ax = axes[1]
    # Take top 30 features and count types
    top_30 = df_feats.head(30)
    type_counts = top_30['Type'].value_counts()
    type_counts.plot(kind='bar', ax=ax, color='teal')
    ax.set_title('Distribution of Types in Top 30 Features')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/wine_full_analysis.png')
    print("\nPlot saved to results/wine_full_analysis.png")

if __name__ == "__main__":
    run_full_experiment()
