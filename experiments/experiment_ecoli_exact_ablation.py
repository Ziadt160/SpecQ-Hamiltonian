import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.utils.data_loader import load_ecoli_reduced as load_ecoli_n4
from src.utils.pauli_utils import generate_pauli_strings
from src.models.exact_sim_classifier import ExactSIMClassifier

# --- Data Loading (Imported from data_loader) ---

# --- Training Helper ---
def train_model(X_train, y_train, epochs=300):
    print("\n--- Training Full Exact SIM Model ---")
    
    # Convert to Tensor (Double prec)
    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    
    model = ExactSIMClassifier(n_qubits=4, n_layers=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
    return model

# --- Importance Calculation ---
def calculate_pauli_importance(model, X_eval):
    """
    Calculate importance for each Pauli string.
    Importance_j = Mean_over_samples( | Classical_j * w_j * Quantum_j | )
    """
    print("\n--- Calculating Pauli String Importance ---")
    model.eval()
    X = torch.tensor(X_eval, dtype=torch.float64)
    
    with torch.no_grad():
        # 1. Classical Features: x^T P_j x  (B, K)
        x_tilde = X + model.b
        classical_features = torch.einsum('bm, kmn, bn -> bk', x_tilde, model.P_tensor, x_tilde)
        
        # 2. Quantum Expectations: <psi | P_j | psi> (K,)
        quantum_expectations = model.qnode(model.circuit_weights)
        if isinstance(quantum_expectations, (list, tuple)):
            quantum_expectations = torch.stack(quantum_expectations)
            
        # 3. Weights w_j (K,)
        w = model.w
        
        # 4. Compute Contribution Terms
        # Contribution_bj = | Classical_bj * w_j * Quantum_j |
        # Broadcast: (B, K) * (K,) * (K,)
        weighted_terms = classical_features * w * quantum_expectations
        abs_contributions = torch.abs(weighted_terms)
        
        # 5. Mean over batch -> Importance (K,)
        importance = torch.mean(abs_contributions, dim=0).numpy()
        
    return importance

# --- Ablation Evaluator ---
def evaluate_subset(model, X_test, y_test, active_indices):
    """
    Evaluates model accuracy using ONLY the Pauli strings in active_indices.
    """
    model.eval()
    X = torch.tensor(X_test, dtype=torch.float64)
    y_true = np.array(y_test)
    
    with torch.no_grad():
        # Re-compute full forward pass components
        x_tilde = X + model.b
        classical_features = torch.einsum('bm, kmn, bn -> bk', x_tilde, model.P_tensor, x_tilde)
        
        quantum_expectations = model.qnode(model.circuit_weights)
        if isinstance(quantum_expectations, (list, tuple)):
            quantum_expectations = torch.stack(quantum_expectations)
            
        combined_weights = model.w * quantum_expectations
        
        # KEY STEP: Mask out non-active indices
        # We achieve this by multiplying inactive terms by 0
        mask = torch.zeros_like(combined_weights)
        mask[active_indices] = 1.0
        
        masked_weights = combined_weights * mask
        
        # Logits
        logits = torch.sum(classical_features * masked_weights, dim=1)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        
        acc = accuracy_score(y_true, preds)
        
    return acc

# --- Main Execution ---
def run_abliation_study():
    # 1. Load Data
    X, y = load_ecoli_n4()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 2. Train Full Model
    model = train_model(X_train, y_train)
    
    # 3. Calculate Importance
    # We use Test set for importance to see what actually drives decisions for unseen data
    # (Or Train set? Using Test set gives 'global feature importance for generalization')
    # Let's use X_train for determining importance to avoid data leakage, 
    # then evaluate ablation on X_test.
    importance_scores = calculate_pauli_importance(model, X_train)
    
    # 4. Rank Strings
    pauli_strings = model.pauli_strings
    # indices sorts ascending, so we take reverse
    sorted_indices = np.argsort(importance_scores)[::-1].copy() 
    
    # Save Ranking
    print("\nTop 10 Most Important Pauli Strings:")
    rank_data = []
    for i in range(len(sorted_indices)):
        idx = sorted_indices[i]
        score = importance_scores[idx]
        p_str = pauli_strings[idx]
        rank_data.append({'Rank': i+1, 'String': p_str, 'Score': score})
        if i < 10:
            print(f"{i+1}. {p_str}: {score:.6f}")
            
    pd.DataFrame(rank_data).to_csv('results/pauli_importance_ranking.csv', index=False)
    
    # 5. Ablation Loop
    print("\n--- Running Ablation Sweep ---")
    k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    results = []
    
    for k in k_values:
        top_k_indices = sorted_indices[:k]
        acc = evaluate_subset(model, X_test, y_test, top_k_indices)
        results.append({'k': k, 'accuracy': acc})
        print(f"Top {k} Strings -> Accuracy: {acc:.4f}")
        
    # 6. Plotting
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/ablation_results.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['k'], results_df['accuracy'], marker='o', linewidth=2)
    plt.xscale('log', base=2)
    plt.xlabel('Number of Pauli Strings (log scale)')
    plt.ylabel('Test Accuracy')
    plt.title('Exact SIM: Accuracy vs Number of Pauli Strings')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xticks(k_values, labels=[str(k) for k in k_values])
    plt.axhline(y=results_df['accuracy'].max(), color='green', linestyle='--', alpha=0.3, label='Max Acc')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/ablation_curve.png')
    print("\nStudy Complete. Results saved to 'results/'")

if __name__ == "__main__":
    run_abliation_study()
