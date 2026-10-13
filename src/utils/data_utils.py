from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .paths import DATA_DIR

# Not shipped with the repository (~300 MB, gitignored).
FRAUD_CSV = DATA_DIR / 'creditcard_2023.csv'
FRAUD_URL = ('https://www.kaggle.com/datasets/nelgiriyewithana/'
             'credit-card-fraud-detection-dataset-2023')


def _load_fraud_sample(n_samples, random_state, csv_path=None):
    """Reads creditcard_2023.csv and draws a class-balanced sample of n_samples rows."""
    path = Path(csv_path) if csv_path is not None else FRAUD_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Credit Card Fraud Detection Dataset 2023 not found at {path}. It is not "
            f"distributed with this repository (~300 MB). Download creditcard_2023.csv "
            f"from {FRAUD_URL} and place it at {FRAUD_CSV}."
        )
    df = pd.read_csv(path)

    # Stratified sample
    fraud = df[df['Class'] == 1].sample(n_samples // 2, random_state=random_state)
    legit = df[df['Class'] == 0].sample(n_samples // 2, random_state=random_state)

    df_subset = pd.concat([fraud, legit]).sample(frac=1, random_state=random_state)

    y = df_subset['Class'].values
    X = df_subset.drop(columns=['id', 'Class']).values
    return X, y


def load_fraud_data(n_samples=5000, n_qubits=6, random_state=42, csv_path=None):
    """
    Unified loader for Credit Card Fraud 2023 dataset.

    Note: PCA and StandardScaler are fit on the whole sample, before any split.
    This is transductive (it uses test *features*, not test labels). Use
    `load_fraud_split` for anything that reports a performance number.
    """
    X, y = _load_fraud_sample(n_samples, random_state, csv_path)

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


def load_fraud_split(n_samples=5000, n_qubits=6, test_size=0.3, random_state=42,
                     stratify=False, csv_path=None):
    """
    Leak-free counterpart to `load_fraud_data`: draws the same balanced sample,
    splits it, then fits PCA and StandardScaler on the training split only.

    The split is `train_test_split(X, y, test_size=test_size,
    random_state=random_state, stratify=y if stratify else None)` on the sampled
    rows, so train/test membership is identical to splitting the output of
    `load_fraud_data` with the same arguments; only the preprocessing differs.

    Returns:
        X_train, X_test, y_train, y_test (np.ndarray)
    """
    X, y = _load_fraud_sample(n_samples, random_state, csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )

    dim = 2**n_qubits
    n_features = X.shape[1]

    if dim <= n_features:
        reduce = PCA(n_components=dim, random_state=random_state).fit(X_train).transform
    else:
        # Pad with zeros if we need more features than available
        def reduce(A):
            A_padded = np.zeros((A.shape[0], dim))
            A_padded[:, :n_features] = A
            return A_padded

    scaler = StandardScaler().fit(reduce(X_train))

    def project(A):
        A_scaled = scaler.transform(reduce(A))
        norms = np.linalg.norm(A_scaled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return A_scaled / norms

    return project(X_train), project(X_test), y_train, y_test
