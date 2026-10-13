import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from src.utils.paths import results_path
from src.utils.seeds import set_seed

from ..generators.spectral_pauli_generator import generate_spectral_pauli_strings
from ..models.exact_sim_classifier import ExactSIMClassifier
from ..utils.data_loader import load_20newsgroups_split


def train_model_k(X_train, y_train, X_test, y_test, pauli_strings, n_qubits, k):
    print(f"\n--- Training Top-{k} (Terms: {len(pauli_strings)}) ---")
    
    # Check dimensions
    if len(pauli_strings) == 0:
        return 0.5 # Random guessing
        
    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    X_te = torch.tensor(X_test, dtype=torch.float64)
    
    model = ExactSIMClassifier(n_qubits=n_qubits, n_layers=3, pauli_strings=pauli_strings)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    # 40 epochs for speed in sweep
    for _ in range(40):
        optimizer.zero_grad()
        out = model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()
            
    # Eval
    model.eval()
    with torch.no_grad():
        preds = (model(X_te).numpy() > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        
    print(f"Top-{k} Accuracy: {acc:.4f}")
    return acc

def run_stress_test():
    
    n_qubits = 6 # 64 dims, 4096 potential terms
    print(f"Loading Data for N={n_qubits}...")
    # TF-IDF, PCA and scaling are fit on the training split only
    X_train, X_test, y_train, y_test = load_20newsgroups_split(n_qubits, test_size=0.3, random_state=42)

    print("Generating Full Spectral Basis Ranking...")
    # Get enough to cover our max k
    max_k = 256
    # All 4096 strings are scored either way; top_k only truncates the ranking.
    spectral_strings, coefs = generate_spectral_pauli_strings(X_train, y_train, n_qubits, top_k=max_k)
    
    k_values = [2, 4, 8, 16, 32, 64, 128, 256]
    accuracies = []
    
    for k in k_values:
        start_k = 0
        current_strings = spectral_strings[start_k:k]
        
        acc = train_model_k(X_train, y_train, X_test, y_test, current_strings, n_qubits, k)
        accuracies.append(acc)
        
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linewidth=2, color='crimson')
    plt.xscale('log', base=2)
    plt.xlabel('Number of Pauli Terms (k)')
    plt.ylabel('Test Accuracy')
    plt.title(f'Spectral Algorithm Stress Test (N={n_qubits})')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xticks(k_values, [str(k) for k in k_values])
    
    plt.savefig(results_path('spectral_stress_test.png'))
    print("Plot saved to results/spectral_stress_test.png")
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame({'k': k_values, 'accuracy': accuracies})
    df.to_csv(results_path('spectral_stress_test.csv'), index=False)

if __name__ == "__main__":
    set_seed()
    run_stress_test()
