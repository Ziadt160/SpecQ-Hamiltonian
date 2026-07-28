import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.utils.seeds import set_seed
set_seed()

def load_fraud_data(n_samples=5000, n_qubits=6, random_state=42):
    """
    Unified loader for Credit Card Fraud 2023 dataset.
    """
    df = pd.read_csv(r'd:\Evoth Labs\SIM-Flipped Models\data\creditcard_2023.csv')
    
    # Stratified sample
    fraud = df[df['Class'] == 1].sample(n_samples // 2, random_state=random_state)
    legit = df[df['Class'] == 0].sample(n_samples // 2, random_state=random_state)
    
    df_subset = pd.concat([fraud, legit]).sample(frac=1, random_state=random_state)
    
    y = df_subset['Class'].values
    X = df_subset.drop(columns=['id', 'Class']).values
    
    dim = 2**n_qubits
    n_features = X.shape[1]
    
    if dim <= n_features:
        pca = PCA(n_components=dim, random_state=random_state)
        X_reduced = pca.fit_transform(X)
    else:
        # Pad with zeros if we need more features than available
        X_reduced = np.zeros((X.shape[0], dim))
        X_reduced[:, :n_features] = X
    
    # Scale and Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)
    
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X_scaled / norms
    
    return X_norm, y
