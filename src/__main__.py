"""Entry point for running the Pokemon AI agent as a module."""
import os
import sys

# Patch per esecuzione diretta (fallback se qualcuno esegue __main__.py direttamente)
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    __package__ = "src"

from .main import main

if __name__ == "__main__":
    main()
