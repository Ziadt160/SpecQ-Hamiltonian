"""Seeding, output paths and data loading."""
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.utils import data_loader, paths
from src.utils.seeds import set_seed


def test_set_seed_is_reproducible():
    set_seed(7)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(7)
    b = (np.random.rand(3), torch.rand(3))
    assert np.array_equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_importing_the_package_does_not_reseed():
    """Library modules used to call set_seed() at import time."""
    # A fresh interpreter, so every module is genuinely imported for the first time.
    code = (
        "import importlib, pkgutil, numpy as np, src\n"
        "np.random.seed(123); expected = np.random.rand(); np.random.seed(123)\n"
        "for m in pkgutil.walk_packages(src.__path__, 'src.'):\n"
        "    importlib.import_module(m.name)\n"
        "assert np.random.rand() == expected, 'an import reseeded NumPy'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=paths.REPO_ROOT)


def test_results_path_is_repo_anchored(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path / "results")
    out = paths.results_path("case_study", "metrics.csv")
    assert out == str(tmp_path / "results" / "case_study" / "metrics.csv")
    assert (tmp_path / "results" / "case_study").is_dir()


def test_row_normalize_handles_zero_rows():
    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert np.allclose(data_loader.row_normalize(X), [[0.6, 0.8], [0.0, 0.0]])


@pytest.mark.skipif(not data_loader.ECOLI_CSV.exists(), reason="E. coli CSV not present")
def test_ecoli_split_selects_features_on_training_labels_only():
    X_tr, X_te, y_tr, y_te = data_loader.load_ecoli_split(n_qubits=4)
    assert X_tr.shape[1] == X_te.shape[1] == 16
    assert len(X_tr) + len(X_te) == len(y_tr) + len(y_te)
    assert set(np.unique(y_tr)) == {0, 1}
    norms = np.linalg.norm(X_tr, axis=1)
    assert np.allclose(norms[norms > 0], 1.0)
