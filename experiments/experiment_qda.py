from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from src.utils.data_loader import load_20newsgroups_split
from src.utils.paths import results_path
from src.utils.seeds import set_seed


def run_qda_comparison():
    print("Loading 20 Newsgroups (N=4)...")
    # TF-IDF, PCA and scaling are fit on the training split only
    X_train, X_test, y_train, y_test = load_20newsgroups_split(n_qubits=4, test_size=0.3, random_state=42)
    
    # QDA
    print("Training QDA...")
    # Reg_param allows for regularization (essential for high dims, but N=4 is 16 dims, so minimal needed)
    qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
    qda.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, qda.predict(X_train))
    test_acc = accuracy_score(y_test, qda.predict(X_test))
    
    print(f"QDA Train Acc: {train_acc:.4f}")
    print(f"QDA Test Acc: {test_acc:.4f}")
    
    # Save
    with open(results_path('qda_results.txt'), 'w') as f:
        f.write("QDA (reg=0.1) Comparison:\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")

if __name__ == "__main__":
    set_seed()
    run_qda_comparison()
