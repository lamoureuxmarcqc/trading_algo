from setuptools import setup, find_packages

setup(
    name="trading-algo",
    version="1.0.0",
    author="Votre Nom",
    description="Système d'analyse et de prédiction boursière avec IA",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "yfinance>=0.2.28",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "trading-algo=main:main",
        ],
    },
)