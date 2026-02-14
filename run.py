#!/usr/bin/env python3
"""
Point d'entrée pour exécuter le trading algorithmique
"""

import sys
import os

# Ajouter src au chemin Python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trading_algo.main import main

if __name__ == "__main__":
    main()
