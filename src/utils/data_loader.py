import os
import urllib.request
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups, load_digits, load_wine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

# Repo root, resolved relative to this file so the loaders work from any cwd.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ECOLI_CSV = os.path.join(REPO_ROOT, 'data', 'EColi_Merged_df.csv')


def row_normalize(X):
    """L2-normalizes each row. Per-sample, so it never leaks across samples."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


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


# ---------------------------------------------------------------------------
# E. Coli
# ---------------------------------------------------------------------------

def load_ecoli_raw():
    """
    Loads the E. Coli gene matrix with no feature selection applied.

    Returns:
        X_genes (np.ndarray): (n_samples, n_genes) raw gene features.
        y (np.ndarray): Binary CTZ resistance labels (1 = resistant).
    """
    path = ECOLI_CSV
    if not os.path.exists(path):
        path = 'data/EColi_Merged_df.csv'
    df = pd.read_csv(path).dropna(subset=['CTZ'])
    y = df['CTZ'].apply(lambda x: 1 if x == 'R' else 0).values
    X_genes = df.iloc[:, 15:].values  # Simplified index
    return X_genes, y


def select_topk_chi2(X_train, X_test, y_train, n_qubits=4):
    """
    Leak-free chi2 feature selection: the selector is fit on the training split
    only, then applied to both splits.

    Args:
        X_train, X_test (np.ndarray): Raw (unreduced) feature matrices.
        y_train (np.ndarray): Training labels only.
        n_qubits (int): Selects k = 2**n_qubits features.

    Returns:
        X_train_r, X_test_r (np.ndarray): Reduced, row-normalized splits.
        selector (SelectKBest): The fitted selector (for inspecting gene indices).
    """
    selector = SelectKBest(chi2, k=2 ** n_qubits).fit(X_train, y_train)
    X_train_r = row_normalize(selector.transform(X_train))
    X_test_r = row_normalize(selector.transform(X_test))
    return X_train_r, X_test_r, selector


def load_ecoli_split(n_qubits=4, test_size=0.3, random_state=42):
    """
    Leak-free E. Coli loader. Splits first, then fits chi2 selection on the
    training split only.

    This is the correct counterpart to `load_ecoli_reduced`, which selects
    features using the labels of the entire dataset (including the test split)
    and therefore reports optimistically biased accuracy.

    Returns:
        X_train, X_test, y_train, y_test (np.ndarray)
    """
    X_genes, y = load_ecoli_raw()
    X_train, X_test, y_train, y_test = train_test_split(
        X_genes, y, test_size=test_size, random_state=random_state
    )
    X_train_r, X_test_r, _ = select_topk_chi2(X_train, X_test, y_train, n_qubits)
    return X_train_r, X_test_r, y_train, y_test


def load_ecoli_reduced(n_qubits=4):
    """
    Loads E. Coli and reduces to 2^n_qubits features.

    WARNING - DATA LEAKAGE: chi2 selection is fit on the full dataset, using the
    labels of samples that later land in the test split. Any test accuracy
    derived from this is optimistically biased. Use `load_ecoli_split` for
    anything that reports a performance number.

    Retained for non-evaluative uses (geometry/visualization analyses) where the
    selection is the object of study rather than an input to a score.
    """
    warnings.warn(
        "load_ecoli_reduced fits chi2 feature selection on the full dataset "
        "(test labels included). Test accuracy from this loader is optimistically "
        "biased - use load_ecoli_split() for reported metrics.",
        UserWarning,
        stacklevel=2,
    )
    X_genes, y = load_ecoli_raw()
    X_reduced = SelectKBest(chi2, k=2 ** n_qubits).fit_transform(X_genes, y)
    return row_normalize(X_reduced), y


# ---------------------------------------------------------------------------
# 20 Newsgroups
# ---------------------------------------------------------------------------

def _fetch_20newsgroups():
    download_20newsgroups_manual()
    categories = ['alt.atheism', 'soc.religion.christian']
    return fetch_20newsgroups(
        subset='all', categories=categories, remove=('headers', 'footers', 'quotes')
    )


def load_20newsgroups_projected(n_qubits):
    """
    Loads 20 Newsgroups and projects to 2^n_qubits dimensions.

    Note: the TF-IDF vocabulary, PCA basis and feature scaling are fit on the
    full corpus before any split. This is transductive (unsupervised - it uses
    test *features* but not test *labels*), so the bias is much milder than the
    E. Coli case. `load_20newsgroups_split` is the leak-free equivalent.
    """
    newsgroups = _fetch_20newsgroups()
    X_tfidf = TfidfVectorizer(stop_words='english', max_features=5000).fit_transform(newsgroups.data)
    X_pca = PCA(n_components=2 ** n_qubits, random_state=42).fit_transform(X_tfidf.toarray())
    X_scaled = StandardScaler().fit_transform(X_pca)
    return row_normalize(X_scaled), newsgroups.target


def load_20newsgroups_split(n_qubits, test_size=0.3, random_state=42):
    """
    Leak-free 20 Newsgroups loader: splits the raw documents first, then fits
    the TF-IDF vocabulary, PCA basis and scaler on the training split only.

    Returns:
        X_train, X_test, y_train, y_test (np.ndarray)
    """
    newsgroups = _fetch_20newsgroups()
    docs_train, docs_test, y_train, y_test = train_test_split(
        newsgroups.data, newsgroups.target, test_size=test_size, random_state=random_state
    )

    vec = TfidfVectorizer(stop_words='english', max_features=5000).fit(docs_train)
    pca = PCA(n_components=2 ** n_qubits, random_state=42).fit(vec.transform(docs_train).toarray())
    scaler = StandardScaler().fit(pca.transform(vec.transform(docs_train).toarray()))

    def _project(docs):
        return row_normalize(scaler.transform(pca.transform(vec.transform(docs).toarray())))

    return _project(docs_train), _project(docs_test), y_train, y_test


def load_20newsgroups_n4():
    """Convenience wrapper: 20 Newsgroups at N=4, returning (X, y, n_qubits)."""
    X, y = load_20newsgroups_projected(4)
    return X, y, 4


# ---------------------------------------------------------------------------
# Digits / Wine
# ---------------------------------------------------------------------------

def load_digits_normalized():
    """Loads 8x8 Digits (64 features -> N=6) and normalizes for quantum states."""
    data = load_digits()
    X_scaled = StandardScaler().fit_transform(data.data)
    return row_normalize(X_scaled), data.target, 6


def load_wine_normalized():
    """Loads Wine dataset and pads to 16 dimensions (N=4)."""
    data = load_wine()
    X_scaled = StandardScaler().fit_transform(data.data)
    X_padded = np.zeros((X_scaled.shape[0], 16))
    X_padded[:, :X_scaled.shape[1]] = X_scaled
    return row_normalize(X_padded), data.target, 4
