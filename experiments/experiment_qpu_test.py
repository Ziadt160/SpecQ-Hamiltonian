import time

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.models.spectral_sim_classifier import SpectralExactSIMClassifier
from src.utils.data_utils import load_fraud_split
from src.utils.paths import results_path
from src.utils.seeds import set_seed


def run_qpu_final_test(device_name='default.qubit', shots=1024, ibmq_token=None, backend=None):
    """
    Final QPU Test Experiment.
    
    Args:
        device_name (str): PennyLane device name (e.g., 'qiskit.ibmq', 'default.qubit').
        shots (int): Number of shots for sampling.
        ibmq_token (str): IBM Quantum API Token.
        backend (str): Specific QPU backend name.
    """
    print(f"Starting Final QPU Test on {device_name}...")
    
    # 1. Load Data (PCA/scaler fit on the training split only)
    X_train, X_test, y_train, y_test = load_fraud_split(
        n_samples=100, n_qubits=4, test_size=0.3, random_state=42) # Smaller sample for QPU test
    
    # 2. Configure Device Kwargs
    dev_kwargs = {}
    if device_name == 'qiskit.ibmq':
        if not ibmq_token:
            print("Error: ibmq_token is required for qiskit.ibmq")
            return
        dev_kwargs['ibmqx_token'] = ibmq_token
        if backend:
            dev_kwargs['backend'] = backend
    
    # 3. Initialize Spectral SIM
    # We use a smaller top_k for the final QPU test to ensure speed
    top_k = 16 
    print(f"Initializing Spectral SIM (top_k={top_k}, shots={shots})...")
    
    model = SpectralExactSIMClassifier(
        n_qubits=4, 
        n_layers=2, 
        top_k=top_k,
        device_name=device_name,
        shots=shots,
        **dev_kwargs
    )
    
    # 4. Train
    print("Training on QPU/Backend...")
    start_t = time.time()
    model.fit(X_train, y_train, epochs=10, lr=0.01) # Fewer epochs for speed
    train_time = time.time() - start_t
    print(f"Training Complete in {train_time:.2f} seconds.")
    
    # 5. Evaluate
    print("Evaluating Test Set...")
    # Use torch.no_grad() for evaluation
    model.eval()
    with torch.no_grad():
        probs_test = model(torch.tensor(X_test, dtype=torch.float64)).numpy()
    auc_test = roc_auc_score(y_test, probs_test)
    
    print("\nFinal QPU Test Results:")
    print(f"Device: {device_name}")
    print(f"Test AUC: {auc_test:.4f}")
    print(f"Train Time: {train_time:.2f}s")
    
    # Save results
    res_df = pd.DataFrame([{
        'Device': device_name,
        'Backend': backend,
        'Shots': shots,
        'TopK': top_k,
        'Test_AUC': auc_test,
        'Train_Time': train_time
    }])
    res_df.to_csv(results_path('case_study_fraud/qpu_test_results.csv'), index=False)
    print("\nResults saved to results/case_study_fraud/qpu_test_results.csv")

if __name__ == '__main__':
    set_seed()
    # Default to local simulator for safety. 
    # User can modify this or pass arguments.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='default.qubit')
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--token', type=str, default=None)
    parser.add_argument('--backend', type=str, default=None)
    args = parser.parse_args()
    
    run_qpu_final_test(
        device_name=args.device,
        shots=args.shots,
        ibmq_token=args.token,
        backend=args.backend
    )
