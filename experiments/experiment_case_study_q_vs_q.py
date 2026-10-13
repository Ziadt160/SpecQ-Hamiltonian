import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import SVC

# Import our unified Spectral SIM Classifier
from src.models.spectral_sim_classifier import SpectralExactSIMClassifier
from src.models.vqc_classifier import NoisyVQC
from src.utils.data_utils import load_fraud_split
from src.utils.paths import results_path
from src.utils.seeds import set_seed


def run_q_vs_q_experiment():
    
    # 1000 samples for reasonable Q simulation times (4 qubits); PCA/scaler fit on train only
    X_train, X_test, y_train, y_test = load_fraud_split(
        n_samples=1000, n_qubits=4, test_size=0.3, random_state=42)
    
    results = []
    
    X_tr_t = torch.tensor(X_train, dtype=torch.float64)
    y_tr_t = torch.tensor(y_train, dtype=torch.float64)
    X_te_t = torch.tensor(X_test, dtype=torch.float64)
    
    # 1. Spectral SIM
    print("\n--- Training Spectral SIM ---")
    start = time.time()
    sim = SpectralExactSIMClassifier(n_qubits=4, n_layers=2, top_k=25)
    sim.fit(X_train, y_train, lr=0.01, epochs=30, verbose=False)
    sim_time = time.time() - start
    
    preds_sim = sim.predict(X_test)
    probs_sim = sim(X_te_t).detach().numpy()
    results.append({
        'Model': 'Spectral SIM',
        'AUC': roc_auc_score(y_test, probs_sim),
        'F1': f1_score(y_test, preds_sim),
        'Time (s)': sim_time
    })
    
    # 2. VQC Variants
    variants = ['StronglyEntangling', 'BasicEntangling']
    for v in variants:
        print(f"\n--- Training VQC ({v}) ---")
        start = time.time()
        # Noise-free NoisyVQC (the same model the noise study evaluates at p=0)
        vqc = NoisyVQC(n_qubits=4, layers=2, variant=v, noise_prob=0.0)
        optimizer = optim.Adam(vqc.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        vqc.train()
        for _ in range(30):
            optimizer.zero_grad()
            out = vqc(X_tr_t)
            loss = criterion(out, y_tr_t)
            loss.backward()
            optimizer.step()
        
        v_time = time.time() - start
        vqc.eval()
        with torch.no_grad():
            probs_vqc = vqc(X_te_t).numpy()
            preds_vqc = (probs_vqc > 0.5).astype(int)
            
        results.append({
            'Model': f'VQC ({v})',
            'AUC': roc_auc_score(y_test, probs_vqc),
            'F1': f1_score(y_test, preds_vqc),
            'Time (s)': v_time
        })
        
    # 3. QSVM
    print("\n--- Training QSVM ---")
    start = time.time()
    # Quantum Kernel using Angle Embedding
    dev_kernel = qml.device("default.qubit", wires=4)
    @qml.qnode(dev_kernel, interface="autograd")
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1[:4], wires=range(4)) # Use first 4 features for angles
        qml.adjoint(qml.AngleEmbedding)(x2[:4], wires=range(4))
        return qml.probs(wires=range(4))
        
    def q_kernel(A, B):
        return np.array([[kernel_circuit(a, b)[0] for b in B] for a in A])
        
    svm = SVC(kernel=q_kernel, probability=True)
    # Fit QSVM on a smaller subset (e.g. 200) since kernel matrix computation scales O(N^2)
    sample_limit = 200
    svm.fit(X_train[:sample_limit], y_train[:sample_limit])
    svm_time = time.time() - start
    
    preds_svm = svm.predict(X_test)
    probs_svm = svm.predict_proba(X_test)[:, 1]
    results.append({
        'Model': 'QSVM',
        'AUC': roc_auc_score(y_test, probs_svm),
        'F1': f1_score(y_test, preds_svm),
        'Time (s)': svm_time
    })
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(results_path('case_study_fraud/q_vs_q_metrics.csv'), index=False)
    print("\n--- Results ---")
    print(df_res)
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.bar(df_res['Model'], df_res['AUC'], alpha=0.7, label='AUC', color='teal')
    plt.bar(df_res['Model'], df_res['F1'], alpha=0.7, label='F1', color='coral', width=0.4)
    plt.ylabel('Score')
    plt.title('Quantum vs Quantum: Performance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_path('case_study_fraud/q_vs_q_performance.png'))
    
if __name__ == '__main__':
    set_seed()
    run_q_vs_q_experiment()
