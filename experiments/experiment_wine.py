from src.utils.data_loader import load_wine_normalized as load_and_preprocess_wine
from src.utils.pauli_utils import generate_pauli_strings, classify_pauli_string
from src.models.sim_classifier import SIMClassifier

def load_and_preprocess_wine():
    """
    Loads Wine dataset and pads to 16 dimensions (4 qubits).
    Target: 3 classes (0, 1, 2).
    """
    data = load_wine()
    X = data.data
    y = data.target
    
    # 1. Standardize original features (important for real data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Pad to 16 dimensions
    n_samples, n_features = X_scaled.shape
    n_qubits = 4
    target_dim = 2**n_qubits
    
    if n_features > target_dim:
        raise ValueError(f"Feature dim {n_features} > Qubit dim {target_dim}")
        
    X_padded = np.zeros((n_samples, target_dim))
    X_padded[:, :n_features] = X_scaled
    
    # 3. Normalize (Quantum State Normalization)
    X_norm = X_padded / np.linalg.norm(X_padded, axis=1, keepdims=True)
    
    return X_norm, y, n_qubits

def run_wine_experiment():
    X, y, n_qubits = load_and_preprocess_wine()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Generate Basis
    all_strings = generate_pauli_strings(n_qubits)
    classified_strings = {s: classify_pauli_string(s) for s in all_strings}
    
    classes = ['C0', 'C1Z', 'C1X', 'C2ZX', 'C2XX', 'CY']
    results = []
    
    print("--- Individual Class Performance ---")
    for cls in classes:
        # Get terms for this class
        basis = [s for s, c in classified_strings.items() if c == cls]
        if not basis: continue
        
        sim = SIMClassifier(pauli_strings=basis, C=10.0, random_state=42)
        sim.fit(X_train, y_train)
        acc = sim.score(X_test, y_test)
        
        print(f"Class {cls} ({len(basis)} terms): Test Acc = {acc:.4f}")
        results.append({
            'Experiment': 'Individual',
            'Added_Class': cls,
            'Basis_Size': len(basis),
            'Test_Acc': acc
        })

    print("\n--- Cumulative Performance (Ablation) ---")
    current_basis = []
    for cls in classes:
        new_terms = [s for s, c in classified_strings.items() if c == cls]
        if not new_terms: continue
        
        current_basis.extend(new_terms)
        
        sim = SIMClassifier(pauli_strings=current_basis, C=10.0, random_state=42)
        sim.fit(X_train, y_train)
        acc = sim.score(X_test, y_test)
        
        print(f"Cumulative (+{cls}): Total Size {len(current_basis)}: Test Acc = {acc:.4f}")
        results.append({
            'Experiment': 'Cumulative',
            'Added_Class': cls,
            'Basis_Size': len(current_basis),
            'Test_Acc': acc
        })
        
    # Save
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/experiment_wine_results.csv', index=False)
    
    plot_wine_results(df)

def plot_wine_results(df):
    plt.figure(figsize=(12, 6))
    
    # Individual
    indiv = df[df['Experiment'] == 'Individual']
    plt.bar(indiv['Added_Class'], indiv['Test_Acc'], alpha=0.6, label='Individual Validity', color='orange')
    
    # Cumulative
    cumul = df[df['Experiment'] == 'Cumulative']
    plt.plot(cumul['Added_Class'], cumul['Test_Acc'], marker='o', linewidth=2, label='Cumulative Accuracy', color='blue')
    
    plt.title('Wine Dataset: Pauli Class Importance')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('results/wine_results.png')
    print("\nPlot saved to results/wine_results.png")

if __name__ == "__main__":
    run_wine_experiment()
