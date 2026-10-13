import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from src.models.exact_sim_classifier import ExactSIMClassifier
from src.utils.data_loader import load_20newsgroups_split
from src.utils.paths import results_path
from src.utils.seeds import set_seed


def train_exact_sim(X_train, y_train, X_test, y_test, n_qubits=4, epochs=100):
    print("\n--- Training Exact SIM (PyTorch + PennyLane) ---")
    print(f"N={n_qubits}, Pauli Strings: {4**n_qubits}")
    
    # Convert to Tensor (Double precision for PennyLane compatibility)
    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    X_te = torch.tensor(X_test, dtype=torch.float64)
    # y_test stays numpy for the sklearn metrics
    
    # Use lightning.qubit if available for speed, else default
    # Note: ExactSIMClassifier defaults to 'default.qubit'
    model = ExactSIMClassifier(n_qubits=n_qubits, n_layers=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    losses = []
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tr) # (B,)
        loss = criterion(outputs, y_tr)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
            
    # Eval
    model.eval()
    with torch.no_grad():
        preds = model(X_te).numpy()
        preds_cls = (preds > 0.5).astype(int)
        acc = accuracy_score(y_test, preds_cls)
        
    print(f"Exact SIM Final Accuracy: {acc:.4f}")
    return acc, losses

def run_experiment():
    n_qubits = 4
    # TF-IDF, PCA and scaling are fit on the training split only
    X_train, X_test, y_train, y_test = load_20newsgroups_split(n_qubits, test_size=0.3, random_state=42)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 2. Exact SIM Only
    acc_exact, losses = train_exact_sim(X_train, y_train, X_test, y_test, n_qubits=n_qubits)
    
    # Plot Convergence
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Exact SIM Loss (N=4)', color='purple', linewidth=2)
    plt.axhline(y=0, color='grey', linestyle='--', alpha=0.3)
    plt.title(f"20 Newsgroups (N=4): Exact SIM (acc={acc_exact:.3f})")
    plt.xlabel('Epoch')
    plt.ylabel('BCELoss')
    plt.legend()
    plt.grid(True)
    output_path = results_path('20newsgroups_exact_n4_only_convergence.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Save Results Text
    with open(results_path('20newsgroups_exact_n4_only_results.txt'), 'w') as f:
        f.write("20 Newsgroups Binary Classification (alt.atheism vs soc.religion.christian)\n")
        f.write(f"Dimensions: {2**n_qubits} (N={n_qubits})\n")
        f.write(f"Exact SIM Accuracy: {acc_exact:.4f}\n")

if __name__ == "__main__":
    set_seed()
    run_experiment()
