#!/usr/bin/env python3
"""
Test des imports
"""

import sys
import os

# Ajouter les chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

print("Test des imports...")
print("=" * 60)

# Test 1: data_extraction
try:
    from data.data_extraction import StockDataExtractor
    print("✅ data_extraction importé")
    
    # Tester l'instanciation
    extractor = StockDataExtractor("AAPL")
    print("✅ StockDataExtractor instancié")
except Exception as e:
    print(f"❌ data_extraction: {e}")

print()

# Test 2: stockmodeltrain
try:
    from models.stockmodeltrain import StockModelTrain
    print("✅ stockmodeltrain importé")
except Exception as e:
    print(f"❌ stockmodeltrain: {e}")

print()

# Test 3: actions_sp500
try:
    from screening.actions_sp500 import screen_sp500
    print("✅ actions_sp500 importé")
except Exception as e:
    print(f"❌ actions_sp500: {e}")

print("=" * 60)
