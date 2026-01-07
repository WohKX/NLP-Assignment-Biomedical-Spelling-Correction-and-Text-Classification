from pathlib import Path
import sys

def ensure_project_root() -> None:
    """Ensure the project root (directory containing config.py) is on sys.path.

    This makes entry scripts runnable both as:
      - python path/to/script.py
      - python -m package.module
    """
    root = Path(__file__).resolve().parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
