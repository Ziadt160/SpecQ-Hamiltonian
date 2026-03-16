import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from analysis_canonical_patterns import load_20newsgroups_projected
from sklearn.model_selection import train_test_split

def run_qda_comparison():
    print("Loading 20 Newsgroups (N=4)...")
    X, y = load_20newsgroups_projected(n_qubits=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
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
    with open('../results/qda_results.txt', 'w') as f:
        f.write(f"QDA (reg=0.1) Comparison:\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")

if __name__ == "__main__":
    run_qda_comparison()
