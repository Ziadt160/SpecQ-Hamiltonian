import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import urllib.request

from ..utils.pauli_utils import generate_pauli_strings
from ..models.exact_sim_classifier import ExactSIMClassifier
from ..utils.data_loader import load_20newsgroups_projected, download_20newsgroups_manual

def download_20newsgroups_manual():
    """
    Manually downloads the 20 Newsgroups dataset to scikit_learn_data to avoid 403 Forbidden.
    """
    # 1. Determine download path
    data_home = os.environ.get('SCIKIT_LEARN_DATA', os.path.join('~', 'scikit_learn_data'))
    data_home = os.path.expanduser(data_home)
    target_dir = os.path.join(data_home, "20news_home")
    os.makedirs(target_dir, exist_ok=True)
    
    archive_path = os.path.join(target_dir, "20news-bydate.tar.gz")
    
    if os.path.exists(archive_path):
        # Check size to ensure it's not a partial failed download
        if os.path.getsize(archive_path) > 1000000:
            print(f"Archive found at {archive_path}, skipping download.")
            return
        
    print(f"Downloading 20 Newsgroups manually to {archive_path}...")
    url = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response, open(archive_path, 'wb') as out_file:
            while True:
                chunk = response.read(1024*1024)
                if not chunk: break
                out_file.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Manual download failed: {e}")

def load_20newsgroups_projected(n_qubits):
    """
    Loads 20 Newsgroups (alt.atheism vs soc.religion.christian) and projects to 2^n_qubits.
    """
    # Ensure data is present
    download_20newsgroups_manual()
    
    categories = ['alt.atheism', 'soc.religion.christian']
    print(f"Loading 20 Newsgroups for N={n_qubits}...")
    try:
        newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'), download_if_missing=True)
    except Exception as e:
        print(f"Fetch failed: {e}. Attempting to proceed if data is cached...")
        # If manual download worked, this should work if 'download_if_missing' finds the file.
        raise e

    X_text = newsgroups.data
    y = newsgroups.target
    
    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_tfidf = vectorizer.fit_transform(X_text)
    
    target_dim = 2**n_qubits
    
    # Simple dense PCA
    X_dense = X_tfidf.toarray()
    pca = PCA(n_components=target_dim, random_state=42)
    X_pca = pca.fit_transform(X_dense)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    
    # Quantum L2 Norm
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X_scaled / norms
    
    return X_norm, y

def get_canonical_pattern(s):
    """
    Strips identities and sorts characters.
    e.g. 'IXYZ' -> 'XYZ'
    e.g. 'ZIIZ' -> 'ZZ'
    """
    chars = sorted([c for c in s if c != 'I'])
    if not chars:
        return "I"
    return "".join(chars)

def train_and_analyze(n_qubits, ax):
    print(f"\n--- Analyzing N={n_qubits} ({2**n_qubits} features) ---")
    
    X, y = load_20newsgroups_projected(n_qubits)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Convert to Tensor
    X_tr = torch.tensor(X_train, dtype=torch.float64)
    y_tr = torch.tensor(y_train, dtype=torch.float64)
    
    # Train Exact SIM
    # N=3 to 6
    # N=6 is 64x64 matrices. 4096 terms. Might be slow per epoch.
    # Reduce epochs heavily for N=6? Or just be patient.
    epochs = 50 if n_qubits < 6 else 30 
    
    model = ExactSIMClassifier(n_qubits=n_qubits, n_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Ep {epoch}: Loss={loss.item():.4f}")
        
    print(f"Final Loss: {loss.item():.4f}")
    
    # --- Extract Effectiveness ---
    with torch.no_grad():
        w_vals = model.w.detach() # (K,)
        expectations = model.qnode(model.circuit_weights)
        if isinstance(expectations, (list, tuple)):
            expectations = torch.stack(expectations)
        
        # Effective Weight = w * <P>
        w_eff = w_vals * expectations
        w_eff_abs = torch.abs(w_eff).numpy()
        
    # Cluster by Pattern
    pauli_strings = model.pauli_strings
    
    df = pd.DataFrame({
        'String': pauli_strings,
        'Importance': w_eff_abs
    })
    df['Pattern'] = df['String'].apply(get_canonical_pattern)
    
    # Sum importance
    pattern_sums = df.groupby('Pattern')['Importance'].sum().sort_values(ascending=False)
    
    # Top 8
    top_patterns = pattern_sums.head(8)
    
    # Plot on given ax
    top_patterns.plot(kind='bar', ax=ax, color='teal', alpha=0.7)
    ax.set_title(f"N={n_qubits} (Dim={2**n_qubits})")
    ax.set_ylabel("Effective Importance")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    return pattern_sums

def run_analysis():
    os.makedirs('results', exist_ok=True)
    
    ns = [3, 4, 5, 6]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Iterate
    for i, n in enumerate(ns):
        train_and_analyze(n, axes[i])
        
    plt.tight_layout()
    plt.savefig('results/20newsgroups_canonical_patterns.png')
    print("\nAnalysis complete. Plot saved to results/20newsgroups_canonical_patterns.png")

if __name__ == "__main__":
    run_analysis()
