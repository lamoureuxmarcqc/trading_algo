#!/usr/bin/env python3
"""
Diagnostic des imports
"""

import sys
import os
import logging

# Try to initialize centralized logging if package available, otherwise fallback
try:
    from trading_algo.logging_config import init_logging
    init_logging(level=os.getenv("LOG_LEVEL", None), logfile=os.getenv("LOG_FILE", None))
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

logger.info("Python path:")
for p in sys.path:
    logger.info(f"  {p}")

logger.info("\n" + "="*60)

# Ajouter src au chemin
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

logger.info(f"\nAjouté au chemin: {src_path}")

logger.info("\n" + "="*60)

# Essayer d'importer
logger.info("\nEssai d'importation...")

try:
    # Test 1: Import depuis data
    from trading_algo.data.data_extraction import StockDataExtractor
    logger.info("✅ StockDataExtractor importé")
    
    # Test 2: Vérifier si find_best_model existe
    import importlib.util
    find_best_model_path = os.path.join(src_path, 'models', 'find_best_model.py')
    logger.info(f"\nChemin de find_best_model.py: {find_best_model_path}")
    logger.info(f"Existe: {os.path.exists(find_best_model_path)}")
    
    if os.path.exists(find_best_model_path):
        spec = importlib.util.spec_from_file_location("find_best_model", find_best_model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'ImprovedLSTMPredictorMultiOutput'):
            logger.info("✅ ImprovedLSTMPredictorMultiOutput trouvé dans le fichier")
        else:
            logger.warning("❌ ImprovedLSTMPredictorMultiOutput NON trouvé")
            
        # Afficher le contenu du fichier
        logger.info(f"\nContenu de find_best_model.py (premières 10 lignes):")
        with open(find_best_model_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 10:
                    logger.info(f"  {i+1}: {line.rstrip()}")
                else:
                    break
                    
except Exception as e:
    logger.error(f"❌ Erreur: {e}", exc_info=True)
