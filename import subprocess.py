import subprocess

# Génère le fichier complet (requirements-dev.txt)
with open("C:\Users\marc\trading_algo\requirements-dev.txt", "w") as f:
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    f.write(result.stdout)

# Génère le fichier minimal (requirements.txt) avec les libs principales
main_libs = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "scipy",
    "tensorflow",
    "keras",
    "plotly",
    "xgboost",
    "catboost",
    "yfinance",
    "ta"
]

installed = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout.splitlines()

with open("C:\Users\marc\trading_algo\requirements.txt", "w") as f:
    for line in installed:
        for lib in main_libs:
            if line.lower().startswith(lib.lower() + "=="):
                f.write(line + "\n")
