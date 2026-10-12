"""Rend le package `app` importable quand pytest tourne depuis ce dossier."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))