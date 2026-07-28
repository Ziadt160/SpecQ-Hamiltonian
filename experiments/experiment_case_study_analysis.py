import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.fwht_sim_classifier import FWHTExactSIMClassifier
from src.utils.data_utils import load_fraud_data

from src.utils.seeds import set_seed
set_seed()

def run_spectral_analysis():
    print("Running Advanced FWHT Spectral Analysis and Classical Comparison on Credit Card Fraud Data...")
    os.makedirs('results/case_study_fraud', exist_ok=True)
    
    # Load data using shared utility (4 qubits = 16 features)
    X, y = load_fraud_data(n_samples=1000,n_qubits=4, random_state=42)
    
    # Normalize features
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Split into Train and Test sets to evaluate Overfitting
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Dataset Size: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
    
    top_k_select = 15
    epochs = 1000
    
    results = {}
    
    # ---------------------------------------------------------
    # 1. FWHT Exact SIM Classifier (Quantum Pytorch version)
    # ---------------------------------------------------------
    print(f"\n[1/4] Training FWHTExactSIMClassifier (Top-{top_k_select} Strings)...")
    sim = FWHTExactSIMClassifier(n_qubits=4, top_k=top_k_select, n_layers=2)
    sim.fit(X_train, y_train, epochs=epochs, lr=0.01, verbose=True)
    
    strings = sim.pauli_strings_
    coefs = sim.spectral_coefs_
    
    y_train_pred_sim = sim.predict(X_train)
    y_test_pred_sim = sim.predict(X_test)
    
    results['FWHT-SIM'] = {
        'Train Acc': accuracy_score(y_train, y_train_pred_sim),
        'Test Acc': accuracy_score(y_test, y_test_pred_sim),
        'Train F1': f1_score(y_train, y_train_pred_sim, zero_division=0),
        'Test F1': f1_score(y_test, y_test_pred_sim, zero_division=0)
    }
    
    # ---------------------------------------------------------
    # 2. Logistic Regression
    # ---------------------------------------------------------
    print("[2/4] Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    y_train_pred_lr = lr_model.predict(X_train)
    y_test_pred_lr = lr_model.predict(X_test)
    
    results['Logistic Regression'] = {
        'Train Acc': accuracy_score(y_train, y_train_pred_lr),
        'Test Acc': accuracy_score(y_test, y_test_pred_lr),
        'Train F1': f1_score(y_train, y_train_pred_lr, zero_division=0),
        'Test F1': f1_score(y_test, y_test_pred_lr, zero_division=0)
    }

    # ---------------------------------------------------------
    # 3. Random Forest
    # ---------------------------------------------------------
    print("[3/4] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X_train, y_train)
    
    y_train_pred_rf = rf_model.predict(X_train)
    y_test_pred_rf = rf_model.predict(X_test)
    
    results['Random Forest'] = {
        'Train Acc': accuracy_score(y_train, y_train_pred_rf),
        'Test Acc': accuracy_score(y_test, y_test_pred_rf),
        'Train F1': f1_score(y_train, y_train_pred_rf, zero_division=0),
        'Test F1': f1_score(y_test, y_test_pred_rf, zero_division=0)
    }

    # ---------------------------------------------------------
    # 4. Support Vector Machine (RBF Kernel)
    # ---------------------------------------------------------
    print("[4/4] Training SVM (RBF)...")
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    
    y_train_pred_svm = svm_model.predict(X_train)
    y_test_pred_svm = svm_model.predict(X_test)
    
    results['SVM (RBF)'] = {
        'Train Acc': accuracy_score(y_train, y_train_pred_svm),
        'Test Acc': accuracy_score(y_test, y_test_pred_svm),
        'Train F1': f1_score(y_train, y_train_pred_svm, zero_division=0),
        'Test F1': f1_score(y_test, y_test_pred_svm, zero_division=0)
    }

    # ==========================================
    # Text Deductions & Analysis Report
    # ==========================================
    orders = [sum(1 for char in s if char != 'I') for s in strings]
    x_count = sum(s.count('X') for s in strings)
    y_count = sum(s.count('Y') for s in strings)
    z_count = sum(s.count('Z') for s in strings)
    
    report_path = 'results/case_study_fraud/spectral_deductions_comparative.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== FWHT Quantum-Inspired Model vs Classical Baselines ===\n\n")
        
        f.write("--- 1. Comprehensive Model Comparison ---\n")
        f.write(f"{'Model Name':<20} | {'Train Acc':<10} | {'Test Acc':<10} | {'Train F1':<10} | {'Test F1':<10}\n")
        f.write("-" * 75 + "\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name:<20} | {metrics['Train Acc']*100:>6.2f}%    | {metrics['Test Acc']*100:>6.2f}%    | {metrics['Train F1']:>8.4f}   | {metrics['Test F1']:>8.4f}\n")
        
        f.write("\n\n--- 2. FWHT Extracted Pauli Springs (Top Features) ---\n")
        f.write(f"FWHT selected the top {top_k_select} Pauli strings natively driving the class variance:\n")
        for i, (s, c) in enumerate(zip(strings, coefs)):
            f.write(f"  {i+1}. {s} : FWHT_Coefficient = {c:+.6f} | Magnitude = {np.abs(c):.6f}\n")
        
        f.write("\n--- 3. FWHT Statistical Distribution ---\n")
        f.write(f"Average Interaction Order (K-locality): {np.mean(orders):.2f} / 4\n")
        f.write(f"Operator Distribution: X={x_count}, Y={y_count}, Z={z_count}, I={sum(s.count('I') for s in strings)}\n\n")

    print(f"\nComparative Analysis saved to {report_path}")
    
    # ==========================================
    # Visualizations
    # ==========================================
    
    # Plot 1: Performance Comparison
    models = list(results.keys())
    test_accs = [results[m]['Test Acc'] for m in models]
    test_f1s = [results[m]['Test F1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, test_accs, width, label='Test Accuracy', color='#5DADE2', edgecolor='black')
    rects2 = ax.bar(x + width/2, test_f1s, width, label='Test F1-Score', color='#58D68D', edgecolor='black')
    
    ax.set_ylabel('Scores')
    ax.set_title('Quantum-Inspired FWHT-SIM vs Classical Models')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig('results/case_study_fraud/models_performance_comparison.png', dpi=300)
    print("Plot saved to results/case_study_fraud/models_performance_comparison.png")
    
    # Plot 2: Spectral Profile
    plt.figure(figsize=(10, 5))
    mags = np.abs(coefs)
    colors = ['#ff9999' if c < 0 else '#66b3ff' for c in coefs]
    
    bars = plt.bar(range(len(coefs)), mags, color=colors, edgecolor='black', alpha=0.8)
    plt.xticks(range(len(strings)), strings, rotation=45, ha='right', fontsize=10)
    plt.xlabel('Selected FWHT Pauli Strings', fontsize=12)
    plt.ylabel('Coefficient Magnitude |c_p|', fontsize=12)
    plt.title('FWHT Pauli Spectral Coefficients (Blue = Pos, Red = Neg)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar, val in zip(bars, coefs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, rotation=90)

    plt.tight_layout()
    plt.savefig('results/case_study_fraud/spectral_importance_profile.png', dpi=300)
    print("Plot saved to results/case_study_fraud/spectral_importance_profile.png")

if __name__ == '__main__':
    run_spectral_analysis()
