import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier
    HAS_XGB = False

from sklearn.metrics import roc_auc_score, f1_score

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our unified Spectral SIM Classifier (Classical version)
from src.models.spectral_sim_classifier import SpectralSIMClassifier

from src.utils.data_utils import load_fraud_data

from src.utils.seeds import set_seed
set_seed()

def run_q_vs_c_experiment():
    os.makedirs('results/case_study_fraud', exist_ok=True)
    
    # Data Loaded via shared utility (already scaled and normalized)
    X_norm, y = load_fraud_data(n_samples=5000, n_qubits=6)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
    
    results = []
    
    # 1. Spectral SIM (Classical Simulation)
    print("\n--- Training Spectral SIM (Classical) ---")
    start = time.time()
    sim = SpectralSIMClassifier(n_qubits=6, top_k=64, C=1.0)
    sim.fit(X_train, y_train)
    q_time = time.time() - start
    
    probs_sim = sim.predict_proba(X_test)[:, 1]
    preds_sim = sim.predict(X_test)
    results.append({
        'Model': 'Spectral SIM',
        'AUC': roc_auc_score(y_test, probs_sim),
        'F1': f1_score(y_test, preds_sim),
        'Time (s)': q_time
    })
    
    # 2. Random Forest
    print("\n--- Training Random Forest ---")
    start = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_time = time.time() - start
    results.append({
        'Model': 'Random Forest',
        'AUC': roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
        'F1': f1_score(y_test, rf.predict(X_test)),
        'Time (s)': rf_time
    })
    
    # 3. XGBoost / Gradient Boosting
    model_name = "XGBoost" if HAS_XGB else "HistGradientBoosting (SKLearn)"
    print(f"\n--- Training {model_name} ---")
    start = time.time()
    xgb = XGBClassifier(random_state=42)
    xgb.fit(X_train, y_train)
    xgb_time = time.time() - start
    results.append({
        'Model': model_name,
        'AUC': roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]),
        'F1': f1_score(y_test, xgb.predict(X_test)),
        'Time (s)': xgb_time
    })
    
    # 4. MLP
    print("\n--- Training MLP ---")
    start = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_time = time.time() - start
    results.append({
        'Model': 'MLP',
        'AUC': roc_auc_score(y_test, mlp.predict_proba(X_test)[:, 1]),
        'F1': f1_score(y_test, mlp.predict(X_test)),
        'Time (s)': mlp_time
    })
    
    # 5. Logistic Regression
    print("\n--- Training Logistic Regression ---")
    start = time.time()
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    lr_time = time.time() - start
    results.append({
        'Model': 'Logistic Regression',
        'AUC': roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]),
        'F1': f1_score(y_test, lr.predict(X_test)),
        'Time (s)': lr_time
    })
    
    df_res = pd.DataFrame(results)
    df_res.to_csv('results/case_study_fraud/q_vs_c_metrics.csv', index=False)
    print("\n--- Results ---")
    print(df_res)
    
    # Visualization
    plt.figure(figsize=(10, 5))
    x = np.arange(len(df_res))
    width = 0.35
    
    plt.bar(x - width/2, df_res['AUC'], width, label='AUC', color='#1f77b4')
    plt.bar(x + width/2, df_res['F1'], width, label='F1', color='#ff7f0e')
    
    plt.xticks(x, df_res['Model'])
    plt.ylabel('Score')
    plt.title('Quantum (SIM) vs Classical Baselines: Performance Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/case_study_fraud/q_vs_c_performance.png')
    
if __name__ == '__main__':
    run_q_vs_c_experiment()
