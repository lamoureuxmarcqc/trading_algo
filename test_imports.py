#!/usr/bin/env python3
"""
Test des imports
"""

import sys
import os
import logging

# Ajouter les chemins
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Try to initialize centralized logging if package available, otherwise fallback
try:
    from trading_algo.logging_config import init_logging
    init_logging(level=os.getenv("LOG_LEVEL", None), logfile=os.getenv("LOG_FILE", None))
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

logger.info("Test des imports...")
logger.info("=" * 60)

# Test 1: data_extraction
try:
    from trading_algo.data.data_extraction import StockDataExtractor
    logger.info("✅ data_extraction importé")
    
    # Tester l'instanciation
    extractor = StockDataExtractor("AAPL")
    logger.info("✅ StockDataExtractor instancié")
except Exception as e:
    logger.error(f"❌ data_extraction: {e}", exc_info=True)

logger.info("")

# Test 2: stockmodeltrain
try:
    from trading_algo.models.stockmodeltrain import StockModelTrain
    logger.info("✅ stockmodeltrain importé")
except Exception as e:
    logger.error(f"❌ stockmodeltrain: {e}", exc_info=True)

logger.info("")

# Test 3: actions_sp500
try:
    from trading_algo.screening.actions_sp500 import screen_sp500
    logger.info("✅ actions_sp500 importé")
except Exception as e:
    logger.error(f"❌ actions_sp500: {e}", exc_info=True)

logger.info("=" * 60)
