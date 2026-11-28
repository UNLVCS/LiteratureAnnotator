"""Repository paths shared by the study_v1 scripts."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main_checkout() -> Path:
    """The main checkout (holds .env.yaml and Vault files), also when run from a git worktree."""
    common = subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=REPO, text=True).strip()
    return (REPO / common).resolve().parent
