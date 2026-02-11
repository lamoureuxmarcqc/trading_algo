#!/usr/bin/env python3
"""
Diagnostic des imports
"""

import sys
import os

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\n" + "="*60)

# Ajouter src au chemin
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

print(f"\nAjouté au chemin: {src_path}")

print("\n" + "="*60)

# Essayer d'importer
print("\nEssai d'importation...")

try:
    # Test 1: Import depuis data
    from data.data_extraction import StockDataExtractor
    print("✅ StockDataExtractor importé")
    
    # Test 2: Vérifier si find_best_model existe
    import importlib.util
    find_best_model_path = os.path.join(src_path, 'models', 'find_best_model.py')
    print(f"\nChemin de find_best_model.py: {find_best_model_path}")
    print(f"Existe: {os.path.exists(find_best_model_path)}")
    
    if os.path.exists(find_best_model_path):
        spec = importlib.util.spec_from_file_location("find_best_model", find_best_model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'ImprovedLSTMPredictorMultiOutput'):
            print("✅ ImprovedLSTMPredictorMultiOutput trouvé dans le fichier")
        else:
            print("❌ ImprovedLSTMPredictorMultiOutput NON trouvé")
            
        # Afficher le contenu du fichier
        print(f"\nContenu de find_best_model.py (premières 10 lignes):")
        with open(find_best_model_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 10:
                    print(f"  {i+1}: {line.rstrip()}")
                else:
                    break
                    
except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
