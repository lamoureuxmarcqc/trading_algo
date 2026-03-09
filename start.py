#!/usr/bin/env python3
"""
Fichier de démarrage simplifié
"""

import subprocess
import sys
import os

# Initialize centralized logging for this CLI script
from trading_algo.logging_config import init_logging

init_logging(level=os.getenv("LOG_LEVEL", None), logfile=os.getenv("LOG_FILE", None))
import logging
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Démarrage du système de trading...")
    logger.info("Choisissez une option:")
    logger.info("1. Screening du S&P 500")
    logger.info("2. Analyser une action")
    logger.info("3. Comparer plusieurs actions")
    logger.info("4. Quitter")
    
    choice = input("\nVotre choix (1-4): ").strip()
    
    if choice == '1':
        subprocess.run([sys.executable, "run.py", "--mode", "screen"])
    elif choice == '2':
        symbol = input("Symbole de l'action (ex: AAPL): ").strip().upper()
        advanced = input("Mode avancé? (o/n): ").strip().lower() == 'o'
        dashboard = input("Créer un dashboard? (o/n): ").strip().lower() == 'o'
        
        cmd = ["python", "run.py", symbol]
        if advanced:
            cmd.append("--advanced")
        if dashboard:
            cmd.append("--dashboard")
        
        subprocess.run(cmd)
    elif choice == '3':
        symbols = input("Symboles à comparer (séparés par des virgules): ").strip()
        subprocess.run([sys.executable, "run.py", symbols, "--mode", "compare"])
    else:
        logger.info("Au revoir!")

if __name__ == "__main__":
    main()
