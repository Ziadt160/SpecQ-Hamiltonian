from src.utils.data_loader import load_digits_normalized as load_and_preprocess_digits
from src.utils.pauli_utils import generate_pauli_strings
from src.models.sim_classifier import SIMClassifier

def load_and_preprocess_digits():
    """
    Loads 8x8 Digits dataset (1797 samples, 64 features).
    Maps to N=6 qubits (2^6 = 64).
    """
    data = load_digits()
    X = data.data
    y = data.target
    
    n_samples, n_features = X.shape
    n_qubits = 6
    if n_features != 2**n_qubits:
        raise ValueError(f"Feature dim {n_features} != 2^{n_qubits}")
        
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check for zero norms to avoid division by zero
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0 # Handle zero vectors if any
    
    X_norm = X_scaled / norms
    
    return X_norm, y, n_qubits

def run_mnist_experiment():
    print("Loading Digits dataset...")
    X, y, n_qubits = load_and_preprocess_digits()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Generating Pauli strings for N={n_qubits} (this may take a moment)...")
    all_strings = generate_pauli_strings(n_qubits)
    print(f"Generated {len(all_strings)} strings.")
    
    # 2. Train Full Model
    # 4096 features is manageable for LogisticRegression, but SGD might be faster if large N.
    # Let's stick to Liblinear for accuracy or switch to SGD.
    print("\n--- Training Full Model (Order 6) ---")
    sim_full = SIMClassifier(pauli_strings=all_strings, C=10.0, random_state=42)
    sim_full.fit(X_train, y_train)
    full_acc = sim_full.score(X_test, y_test)
    print(f"Full Basis Test Accuracy: {full_acc:.4f}")
    
    # Analyze Top Features
    weights = sim_full.classifier.coef_
    mean_weights = np.mean(np.abs(weights), axis=0)
    
    feat_data = []
    for i, s in enumerate(all_strings):
        weight_order = sum(1 for c in s if c != 'I')
        # Simple type (e.g. ZZ, XZ)
        chars = sorted([c for c in s if c != 'I'])
        s_type = "".join(chars)
        
        feat_data.append({
            'String': s,
            'Order': weight_order,
            'Type': s_type,
            'Importance': mean_weights[i]
        })
        
    df_feats = pd.DataFrame(feat_data).sort_values('Importance', ascending=False)
    print("\n--- Top 10 Features ---")
    print(df_feats.head(10))
    df_feats.to_csv('results/mnist_full_features.csv', index=False)
    
    # 3. Ablation by Order
    print("\n--- Ablation by Interaction Order ---")
    results = []
    
    for w in range(n_qubits + 1):
        # Subset basis
        # Optimization: Don't re-instantiate SIM every time if we can just select columns?
        # Creating SIM takes strings.
        
        current_basis = [s for s in all_strings if sum(1 for c in s if c != 'I') <= w]
        # Note: This is "Cumulative" by order (Orders 0..w included)
        
        if not current_basis:
            continue
            
        print(f"Training Order <= {w} (Size {len(current_basis)})...")
        
        # Use SGD for faster feedback loop if needed, but here 4096 is fine.
        sim = SIMClassifier(pauli_strings=current_basis, C=10.0, random_state=42)
        sim.fit(X_train, y_train)
        acc = sim.score(X_test, y_test)
        
        print(f"  -> Test Acc: {acc:.4f}")
        results.append({
            'Max_Order': w,
            'Basis_Size': len(current_basis),
            'Test_Acc': acc
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv('results/mnist_order_ablation.csv', index=False)
    
    plot_mnist_results(df_res, df_feats)

def plot_mnist_results(df_res, df_feats):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1
    ax = axes[0]
    ax.plot(df_res['Max_Order'], df_res['Test_Acc'], marker='o', linewidth=2, color='crimson')
    ax.set_title('MNIST (8x8): Accuracy vs Max Interaction Order')
    ax.set_xlabel('Max Pauli Weight')
    ax.set_ylabel('Test Accuracy')
    ax.grid(True)
    
    # Plot 2: Top Feature Types
    ax = axes[1]
    top_50 = df_feats.head(50)
    top_50['Type'].value_counts().plot(kind='bar', ax=ax, color='darkblue')
    ax.set_title('Top 50 Feature Types')
    
    plt.tight_layout()
    plt.savefig('results/mnist_analysis.png')
    print("Plot saved to results/mnist_analysis.png")

if __name__ == "__main__":
    run_mnist_experiment()
