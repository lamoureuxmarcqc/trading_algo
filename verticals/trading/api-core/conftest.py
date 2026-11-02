"""Rend `app` et `trading_algo` importables pour pytest."""
import sys
from pathlib import Path

HERE = Path(__file__).parent            # verticals/trading/api-core
VERTICAL = HERE.parent                  # verticals/trading

# `app` (dans api-core/)
sys.path.insert(0, str(HERE))
# `trading_algo` (dans verticals/trading/trading_algo/)
sys.path.insert(0, str(VERTICAL))