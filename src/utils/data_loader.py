import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups, load_digits, load_wine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

def download_20newsgroups_manual():
    """Manually downloads the 20 Newsgroups dataset."""
    data_home = os.environ.get('SCIKIT_LEARN_DATA', os.path.join('~', 'scikit_learn_data'))
    data_home = os.path.expanduser(data_home)
    target_dir = os.path.join(data_home, "20news_home")
    os.makedirs(target_dir, exist_ok=True)
    
    archive_path = os.path.join(target_dir, "20news-bydate.tar.gz")
    if os.path.exists(archive_path) and os.path.getsize(archive_path) > 1000000:
        return archive_path
        
    url = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(archive_path, 'wb') as out_file:
        out_file.write(response.read())
    return archive_path

def load_20newsgroups_projected(n_qubits):
    """Loads 20 Newsgroups and projects to 2^n_qubits."""
    download_20newsgroups_manual()
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    X_tfidf = TfidfVectorizer(stop_words='english', max_features=5000).fit_transform(newsgroups.data)
    X_pca = PCA(n_components=2**n_qubits, random_state=42).fit_transform(X_tfidf.toarray())
    X_scaled = StandardScaler().fit_transform(X_pca)
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X_scaled / norms, newsgroups.target

def load_ecoli_reduced(n_qubits=4):
    """Loads E. Coli and reduces to 2^n_qubits."""
    path = r'd:\Evoth Labs\SIM-Flipped Models\data\EColi_Merged_df.csv'
    if not os.path.exists(path): path = 'data/EColi_Merged_df.csv'
    df = pd.read_csv(path).dropna(subset=['CTZ'])
    y = df['CTZ'].apply(lambda x: 1 if x == 'R' else 0).values
    X_genes = df.iloc[:, 15:].values # Simplified index
    X_reduced = SelectKBest(chi2, k=2**n_qubits).fit_transform(X_genes, y)
    norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X_reduced / norms, y

def load_digits_normalized():
    """Loads MNIST digits and normalizes for quantum states."""
    data = load_digits()
    X_scaled = StandardScaler().fit_transform(data.data)
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X_scaled / norms, data.target, 6

def load_wine_normalized():
    """Loads Wine dataset and pads to 16 dimensions."""
    data = load_wine()
    X_scaled = StandardScaler().fit_transform(data.data)
    X_padded = np.zeros((X_scaled.shape[0], 16))
    X_padded[:, :X_scaled.shape[1]] = X_scaled
    norms = np.linalg.norm(X_padded, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X_padded / norms, data.target, 4
