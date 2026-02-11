#!/usr/bin/env python3
"""
Fichier de démarrage simplifié
"""

import subprocess
import sys

def main():
    print("🚀 Démarrage du système de trading...")
    print("Choisissez une option:")
    print("1. Screening du S&P 500")
    print("2. Analyser une action")
    print("3. Comparer plusieurs actions")
    print("4. Quitter")
    
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
        print("Au revoir!")

if __name__ == "__main__":
    main()
