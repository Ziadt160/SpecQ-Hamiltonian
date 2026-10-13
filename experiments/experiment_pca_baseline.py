from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.utils.data_loader import load_20newsgroups_split
from src.utils.paths import results_path
from src.utils.seeds import set_seed


def run_pca_baseline():
    print("Loading 20 Newsgroups (N=4)...")
    # X is already PCA reduced to 16 dims and Normalized; TF-IDF, PCA and
    # scaling are fit on the training split only
    X_train, X_test, y_train, y_test = load_20newsgroups_split(n_qubits=4, test_size=0.3, random_state=42)

    print("Training Logistic Regression on Raw PCA Features (16-dim)...")
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, lr.predict(X_train))
    test_acc = accuracy_score(y_test, lr.predict(X_test))
    
    print(f"PCA-LR Train Acc: {train_acc:.4f}")
    print(f"PCA-LR Test Acc: {test_acc:.4f}")
    
    with open(results_path('pca_baseline.txt'), 'w') as f:
        f.write("Logistic Regression on PCA (16-dim):\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")

if __name__ == "__main__":
    set_seed()
    run_pca_baseline()
