"""Repository-anchored locations, so scripts behave identically from any working directory."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def results_path(*parts):
    """
    Path to an output file under results/, creating its parent directory.

    Experiments previously wrote to the relative path 'results/...', so running
    one from inside experiments/ silently produced a second results tree there.
    """
    path = RESULTS_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
