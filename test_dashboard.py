#!/usr/bin/env python3
"""
Test du module dashboard
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ajouter le répertoire src au chemin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_algo.visualization.dashboard import TradingDashboard, MiniDashboard

def create_sample_data():
    """Crée des données d'exemple pour tester le dashboard"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Prix avec tendance et volatilité
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Créer un DataFrame avec des colonnes OHLCV
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100),
        'RSI': np.random.uniform(30, 70, 100),
        'MACD': np.random.uniform(-2, 2, 100),
        'MACD_Signal': np.random.uniform(-2, 2, 100),
        'MACD_Histogram': np.random.uniform(-1, 1, 100),
        'ATR': np.random.uniform(1, 5, 100),
        'SMA_20': prices.rolling(20).mean(),
        'SMA_50': prices.rolling(50).mean(),
        'SMA_200': prices.rolling(200).mean(),
        'BB_Upper': prices.rolling(20).mean() + (prices.rolling(20).std() * 2),
        'BB_Lower': prices.rolling(20).mean() - (prices.rolling(20).std() * 2),
        'BB_Middle': prices.rolling(20).mean(),
        'Volume_SMA': np.random.randint(1000000, 5000000, 100).rolling(20).mean(),
        'Volume_Ratio': np.random.uniform(0.5, 2, 100)
    }, index=dates)
    
    return df

def test_trading_dashboard():
    """Test du dashboard principal"""
    print("🧪 Test du TradingDashboard")
    print("=" * 50)
    
    symbol = "AAPL"
    current_price = 175.50
    
    # Créer le dashboard
    dashboard = TradingDashboard(symbol, current_price)
    
    # Créer des données d'exemple
    technical_data = create_sample_data()
    
    # Données d'aperçu
    overview = {
        'name': 'Apple Inc.',
        'market_cap': '2.8T',
        'pe_ratio': '28.5',
        'dividend_yield': '0.55%',
        'sector': 'Technology',
        '52_week_high': '199.62',
        '52_week_low': '124.17'
    }
    
    # Prévisions
    last_date = technical_data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(30)]
    future_prices = [current_price * (1 + np.random.normal(0.001, 0.02)) for _ in range(30)]
    
    predictions = {
        'future_prices': future_prices,
        'future_dates': future_dates,
        'return_1d': 0.5,
        'return_30d': 2.5,
        'return_90d': 5.0
    }
    
    # Données macro
    macro_data = {
        'interest_rate': 4.5,
        'inflation': 2.1,
        'unemployment': 3.8
    }
    
    # Sentiment marché
    market_sentiment = {
        'overall_sentiment': {
            'score': 65,
            'label': 'Bullish'
        }
    }
    
    # Charger les données dans le dashboard
    dashboard.load_data(
        overview=overview,
        technical_data=technical_data,
        predictions=predictions,
        macro_data=macro_data,
        market_sentiment=market_sentiment,
        score=7.2,
        recommendation="ACHAT 🟢"
    )
    
    # Créer le dashboard
    print("\n📊 Création du dashboard principal...")
    fig = dashboard.create_main_dashboard(save_path="test_dashboards")
    
    if fig:
        print("✅ Dashboard créé avec succès!")
        
        # Afficher l'aperçu rapide
        print("\n📋 Aperçu rapide:")
        overview_df = dashboard.create_quick_overview()
        if overview_df is not None:
            print(overview_df.to_string(index=False))
        
        # Afficher le résumé technique
        print("\n📈 Résumé technique:")
        tech_summary = dashboard.create_technical_summary()
        if tech_summary is not None:
            print(tech_summary.to_string(index=False))
        
        # Sauvegarder le dashboard HTML
        print(f"\n💾 Dashboard sauvegardé dans: test_dashboards/")
        
    else:
        print("❌ Échec de la création du dashboard")
    
    return dashboard

def test_mini_dashboard():
    """Test du mini dashboard"""
    print("\n\n🧪 Test du MiniDashboard")
    print("=" * 50)
    
    symbol = "MSFT"
    mini_dashboard = MiniDashboard(symbol)
    
    # Créer des données
    technical_data = create_sample_data()
    last_date = technical_data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(10)]
    future_prices = [180 * (1 + np.random.normal(0.001, 0.02)) for _ in range(10)]
    
    predictions = {
        'future_prices': future_prices,
        'future_dates': future_dates
    }
    
    # Créer la vue compacte
    fig = mini_dashboard.create_compact_view(technical_data, predictions)
    
    if fig:
        # Sauvegarder la figure
        os.makedirs("test_dashboards", exist_ok=True)
        fig.write_html(f"test_dashboards/{symbol}_mini_dashboard.html")
        print(f"✅ Mini dashboard sauvegardé: test_dashboards/{symbol}_mini_dashboard.html")
    
    return mini_dashboard

def test_comparison_dashboard():
    """Test du dashboard de comparaison"""
    print("\n\n🧪 Test du dashboard de comparaison")
    print("=" * 50)
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    # Créer des données pour chaque symbole
    data_dict = {}
    for symbol in symbols:
        data_dict[symbol] = create_sample_data()
    
    # Importer la fonction
    from trading_algo.visualization.dashboard import create_comparison_dashboard
    
    # Créer le dashboard de comparaison
    fig = create_comparison_dashboard(symbols, data_dict)
    
    if fig:
        # Sauvegarder
        os.makedirs("test_dashboards", exist_ok=True)
        fig.write_html("test_dashboards/comparison_dashboard.html")
        print("✅ Dashboard de comparaison sauvegardé: test_dashboards/comparison_dashboard.html")
    
    return fig

def main():
    """Fonction principale de test"""
    print("🚀 Test des modules de dashboard")
    print("=" * 60)
    
    try:
        # Test 1: Dashboard principal
        dashboard = test_trading_dashboard()
        
        # Test 2: Mini dashboard
        mini_dashboard = test_mini_dashboard()
        
        # Test 3: Dashboard de comparaison
        comparison_dashboard = test_comparison_dashboard()
        
        print("\n" + "=" * 60)
        print("✅ Tous les tests ont été exécutés avec succès!")
        print("\n📁 Les dashboards ont été sauvegardés dans le dossier 'test_dashboards/'")
        print("📄 Ouvrez les fichiers .html dans votre navigateur pour visualiser les dashboards")
        
    except Exception as e:
        print(f"❌ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
