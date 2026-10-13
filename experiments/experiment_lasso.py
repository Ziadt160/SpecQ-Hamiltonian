import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.generators.spectral_pauli_generator import generate_spectral_pauli_strings
from src.utils.data_loader import load_20newsgroups_projected
from src.utils.paths import results_path
from src.utils.pauli_utils import get_pauli_tensor
from src.utils.seeds import set_seed


def run_lasso_comparison():
    print("Loading 20 Newsgroups (N=4)...")
    from sklearn.model_selection import train_test_split
    X, y = load_20newsgroups_projected(n_qubits=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 1. Generate Full Pauli Basis Features
    print("Generating Full Pauli Features (256 terms)...")
    full_pauli_strings, _, _ = generate_spectral_pauli_strings(X_train, y_train, 4) # Get all sorted
    # Actually, let's just use the order returned by spectral to make comparison easier, 
    # but Lasso needs to see *all* features to select them.
    
    # To be fair, Lasso takes raw input transformed by ALL 256 Pauli strings.
    # X_pauli = (x^T P x) for all P.
    
    def transform_to_pauli_features(X_data, pauli_strings):
        # This is computationally heavy if optimization isn't used, but for N=4 (256) it's fine.
        n_samples = X_data.shape[0]
        n_features = len(pauli_strings)
        X_transformed = np.zeros((n_samples, n_features))
        
        for i, p_str in enumerate(pauli_strings):
            P_tensor = get_pauli_tensor(p_str)
            # Efficient quadratic form: x^T M x. Since P is diagonal in Z-basis? No, general P.
            # But wait, our ExactSIM uses efficient calculation.
            # For classical Lasso, we can just compute it explicitly. 
            # Note: The Pauli matrices are 2^N x 2^N.
            # X_data is (samples, 2^N).
            # x^T P x
            # Let's assume P is real/Hermitian.
            P_matrix = P_tensor # Getting matrix form
            
            # (Samples, 1, Dim) @ (Dim, Dim) @ (Samples, Dim, 1) -> (Samples, 1, 1)
            # Use einsum for batch processing: x_i^a P_ab x_i^b
            # X_data is real. Expectation of P might be complex? 
            # Pauli matrices are Hermitian, expectation x^T P x is real.
            
            # Optimization: 
            # result = sum( (X @ P) * X, axis=1 )
            # X @ P might be complex.
            
            XP = X_data @ P_matrix
            # element-wise multiply and sum
            val = np.sum(X_data * XP, axis=1)
            X_transformed[:, i] = np.real(val)
            
        return X_transformed

    print("Computing Pauli feature expansion (this may take a moment)...")
    # Use the same list as spectral to keep indices aligned
    all_pauli_strings = full_pauli_strings

    X_train_pauli = transform_to_pauli_features(X_train, all_pauli_strings)
    X_test_pauli = transform_to_pauli_features(X_test, all_pauli_strings)
    
    scaler = StandardScaler()
    X_train_pauli = scaler.fit_transform(X_train_pauli)
    X_test_pauli = scaler.transform(X_test_pauli)
    
    # 2. Train Lasso (L1)
    print("Training Lasso (L1 Logistic Regression)...")
    # Smaller C = stronger regularization (fewer features). C is fixed at 0.5, not
    # tuned; the number of selected features is whatever that C yields.
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42)
    lasso.fit(X_train_pauli, y_train)
    
    train_acc = accuracy_score(y_train, lasso.predict(X_train_pauli))
    test_acc = accuracy_score(y_test, lasso.predict(X_test_pauli))
    
    # 3. Analyze Selected Features
    coefs = lasso.coef_.flatten()
    nonzero_indices = np.where(np.abs(coefs) > 1e-5)[0]
    n_selected = len(nonzero_indices)
    
    print(f"Lasso Selected Features: {n_selected}")
    print(f"Lasso Train Acc: {train_acc:.4f}")
    print(f"Lasso Test Acc: {test_acc:.4f}")
    
    # 4. Compare with Spectral Ranking
    print("\nComparing Lasso selection with Spectral Ranking...")
    # Get the indices of the selected features in the spectral list (which is sorted by spectral energy)
    # The 'all_pauli_strings' list IS the spectral sorted list.
    # So 'nonzero_indices' directly tells us the rank of Lasso selections.
    
    # Example: If Lasso selects index 0, 1, 2... it agrees perfectly with Spectral.
    # If it selects index 200, it disagrees.
    
    selected_ranks = np.sort(nonzero_indices)
    overlap_top_k = len([i for i in nonzero_indices if i < n_selected])
    overlap_pct = (overlap_top_k / n_selected) * 100

    # Chance level. Lasso can only pick non-constant columns (dead strings are
    # identically zero, and the identity is |x|^2 = 1 after row normalisation),
    # so the fair baseline is a random subset of the same size drawn from those.
    selectable = np.where(scaler.var_ > 1e-12)[0]
    chance_overlap_pct = 100 * np.mean(selectable < n_selected)
    chance_mean_rank = np.mean(selectable)

    print(f"Lasso chose {n_selected} features.")
    print(f"Of these, {overlap_top_k} were also in the Spectral Top-{n_selected}.")
    print(f"Overlap: {overlap_pct:.2f}%  (chance: {chance_overlap_pct:.2f}% for a random "
          f"{n_selected} of the {len(selectable)} non-constant features)")

    # Save results
    with open(results_path('lasso_comparison.txt'), 'w') as f:
        f.write("Lasso C=0.5 Results:\n")
        f.write(f"Selected Features: {n_selected}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Overlap with Spectral Top-{n_selected}: {overlap_pct:.2f}%\n")
        f.write(f"Average Rank of Lasso selections: {np.mean(selected_ranks):.1f} (Lower is better/closer to spectral)\n")
        f.write(f"Chance level (random {n_selected} of the {len(selectable)} non-constant "
                f"features): overlap {chance_overlap_pct:.2f}%, average rank {chance_mean_rank:.1f}\n")

if __name__ == "__main__":
    set_seed()
    run_lasso_comparison()
