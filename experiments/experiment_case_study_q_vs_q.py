import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our unified Spectral SIM Classifier
from src.models.spectral_sim_classifier import SpectralExactSIMClassifier

from src.utils.data_utils import load_fraud_data
from src.models.vqc_classifier import VQCPennyLane

from src.utils.seeds import set_seed
set_seed()

def run_q_vs_q_experiment():
    os.makedirs('results/case_study_fraud', exist_ok=True)
    
    X, y = load_fraud_data(n_samples=1000, n_qubits=4) # 1000 samples for reasonable Q simulation times (4 qubits)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    results = []
    
    X_tr_t = torch.tensor(X_train, dtype=torch.float64)
    y_tr_t = torch.tensor(y_train, dtype=torch.float64)
    X_te_t = torch.tensor(X_test, dtype=torch.float64)
    y_te_t = torch.tensor(y_test, dtype=torch.float64)
    
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
        vqc = VQCPennyLane(n_qubits=4, layers=2, variant=v)
        optimizer = optim.Adam(vqc.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        vqc.train()
        for ep in range(30):
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
    df_res.to_csv('results/case_study_fraud/q_vs_q_metrics.csv', index=False)
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
    plt.savefig('results/case_study_fraud/q_vs_q_performance.png')
    
if __name__ == '__main__':
    run_q_vs_q_experiment()
