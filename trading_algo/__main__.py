#!/usr/bin/env python3
"""
Point d'entrée principal du système d'analyse boursière
"""
import sys
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def print_banner():
    banner = r"""
╔══════════════════════════════════════════════════════════════╗
║ SYSTÈME D'ANALYSE BOURSIÈRE AVEC IA ║
║ 🚀 Trading Algorithmique Avancé 🚀 ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def import_modules():
    """Importe tous les modules nécessaires"""
    modules = {}
    
    try:
        from trading_algo.data.data_extraction import StockDataExtractor, get_stock_overview, MacroDataExtractor
        modules['StockDataExtractor'] = StockDataExtractor
        modules['get_stock_overview'] = get_stock_overview
        modules['MacroDataExtractor'] = MacroDataExtractor
        print("✅ Module data_extraction importé")
    except ImportError as e:
        print(f"❌ Erreur data_extraction: {e}")
        return None
    
    try:
        from trading_algo.models.stockmodeltrain import StockModelTrain
        modules['StockModelTrain'] = StockModelTrain
        print("✅ Module stockmodeltrain importé")
    except ImportError as e:
        print(f"❌ Erreur stockmodeltrain: {e}")
        return None
    
    try:
        from trading_algo.models.stockpredictor import StockPredictor
        modules['StockPredictor'] = StockPredictor
        print("✅ Module stockpredictor importé")
    except ImportError as e:
        print(f"⚠️ Module stockpredictor non disponible: {e}")
        modules['StockPredictor'] = None
    
    try:
        from trading_algo.visualization.dashboard import TradingDashboard, create_comparison_dashboard
        modules['TradingDashboard'] = TradingDashboard
        modules['create_comparison_dashboard'] = create_comparison_dashboard
        print("✅ Module dashboard importé")
    except ImportError as e:
        print(f"⚠️ Module dashboard non disponible: {e}")
        modules['TradingDashboard'] = None
        modules['create_comparison_dashboard'] = None
    
    # Module screening optionnel
    try:
        from trading_algo.screening.actions_sp500 import screen_sp500, get_sp500_symbols
        modules['screen_sp500'] = screen_sp500
        modules['get_sp500_symbols'] = get_sp500_symbols
        print("✅ Module actions_sp500 importé")
    except ImportError:
        # Silencieux si pas présent
        modules['screen_sp500'] = None
        modules['get_sp500_symbols'] = None
    
    return modules

def run_screening(modules):
    """Exécute le screening du S&P 500"""
    print("\n🔍 SCREENING DU S&P 500")
    print("=" * 60)
    
    screen_sp500 = modules.get('screen_sp500')
    if screen_sp500 is None:
        print("❌ Module de screening non disponible")
        return
    
    try:
        results = screen_sp500(
            lookback_years=3,
            min_probability=0.65,
            max_stocks=15
        )
        
        if results['success']:
            stocks = results['stocks']
            
            print(f"\n📊 RÉSULTATS:")
            print(f" 🎯 Précision du modèle: {results['metrics']['accuracy']:.3f}")
            print(f" 📈 Stocks recommandés: {len(stocks)}")
            
            if stocks:
                print(f"\n🏆 ACTIONS RECOMMANDÉES:")
                for i, stock in enumerate(stocks[:10], 1):
                    print(f" {i}. {stock['symbol']}:")
                    print(f" Prix: ${stock['current_price']:.2f}")
                    print(f" Probabilité: {stock['buy_probability']:.1%}")
                    print(f" Signal: {stock['signal_strength']}")
                    if stock.get('rsi'):
                        rsi = stock['rsi']
                        status = "SURACHAT ⚠️" if rsi > 70 else "SURVENTE ✅" if rsi < 30 else "NEUTRE"
                        print(f" RSI: {rsi:.1f} ({status})")
                    print()
            
            print(f"\n💾 Résultats sauvegardés dans: {results['output_file']}")
            print("💡 Analysez une action avec: trading-algo SYMBOLE --advanced")
        else:
            print(f"❌ Erreur: {results['error']}")
            
    except Exception as e:
        print(f"❌ Erreur screening: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)

def compare_stocks(symbols, period, modules):
    """Compare plusieurs actions"""
    print(f"\n📊 COMPARAISON DE {len(symbols)} ACTIONS")
    print("=" * 60)
    
    try:
        data_dict = {}
        extractor_class = modules['StockDataExtractor']
        
        for symbol in symbols:
            print(f"\n📈 Récupération des données pour {symbol}...")
            extractor = extractor_class(symbol)
            data = extractor.get_historical_data(period=period)
            
            if not data.empty:
                data_dict[symbol] = data
                print(f" ✅ {len(data)} périodes récupérées")
            else:
                print(f" ❌ Données non disponibles")
        
        if len(data_dict) > 1 and modules.get('create_comparison_dashboard'):
            print(f"\n📊 Création du dashboard de comparaison...")
            fig = modules['create_comparison_dashboard'](list(data_dict.keys()), data_dict)
            
            if fig:
                os.makedirs("dashboards", exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"dashboards/comparison_{'_'.join(symbols)}_{timestamp}.html"
                fig.write_html(filename)
                print(f"\n✅ Dashboard sauvegardé: {filename}")
                print("📄 Ouvrez ce fichier dans votre navigateur")
        else:
            print("❌ Pas assez de données ou module dashboard non disponible")
            
    except Exception as e:
        print(f"❌ Erreur comparaison: {e}")

def analyze_stock(symbol, period, mode, advanced, create_dashboard, modules):
    """Analyse une action"""
    print(f"\n🔍 Analyse pour {symbol} - période: {period}")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        if advanced and modules.get('StockPredictor'):
            print("🧠 Utilisation du module avancé StockPredictor...")
            predictor = modules['StockPredictor'](symbol, period)
        else:
            print("🤖 Utilisation du module de base StockModelTrain...")
            predictor = modules['StockModelTrain'](symbol, period)
        
        if mode == "train":
            print("\n🤖 Entraînement du modèle...")
            success = predictor.train(
                lookback_days=60,
                epochs=30
            )
            if success:
                print("✅ Modèle entraîné avec succès!")
            else:
                print("❌ Échec de l'entraînement")
            return
        
        print(f"\n🔍 Analyse de {symbol}...")
        
        if advanced and modules.get('StockPredictor'):
            results = predictor.analyze_stock_advanced()
        else:
            results = predictor.analyze_model_stock()
        
        if 'error' in results:
            print(f"❌ Erreur: {results['error']}")
            return
        
        print(f"\n📊 RÉSULTATS POUR {symbol}:")
        print(f" 📈 Prix actuel: ${results['current_price']:.2f}")
        print(f" 🎯 Score de trading: {results['trading_score']}/10")
        print(f" 💡 Recommandation: {results['recommendation']}")
        
        predictions = results.get('predictions', {})
        if predictions:
            print("\n 🔮 Prédictions de prix:")
            for horizon in ['1d', '5d', '10d', '20d', '30d', '90d']:
                if horizon in predictions and predictions[horizon] is not None:
                    pred = predictions[horizon]
                    change_pct = ((pred - results['current_price']) / results['current_price'] * 100)
                    print(f" {horizon}: ${pred:.2f} ({change_pct:+.2f}%)")
        
        # Affichage des métriques de risque si disponibles
        if 'risk_metrics' in results and advanced:
            rm = results['risk_metrics']
            print("\n⚠️ MÉTRIQUES DE RISQUE:")
            if 'sharpe_ratio' in rm:
                print(f" Sharpe Ratio: {rm['sharpe_ratio']:.2f}")
            if 'max_drawdown' in rm:
                print(f" Max Drawdown: {rm['max_drawdown'] * 100:.1f}%")
            if 'value_at_risk' in rm:
                print(f" Value at Risk (95%): {rm['value_at_risk'] * 100:.1f}%")
            if 'beta' in rm:
                print(f" Bêta: {rm['beta']:.2f}")
            
            if 'stop_loss_levels' in rm:
                print("\n Niveaux de gestion du risque:")
                for horizon in ['1d', '5d', '10d', '20d', '30d', '90d']:
                    if horizon in rm['stop_loss_levels']:
                        stop = rm['stop_loss_levels'][horizon]
                        tp = rm['take_profit_levels'].get(horizon, 'N/A')
                        rr = rm['risk_reward_ratios'].get(horizon, 'N/A')
                        pos = rm['suggested_position_sizes'].get(horizon, 'N/A')
                        print(f" {horizon}: Stop ${stop:.2f}, Target ${tp:.2f}, R/R {rr:.2f}, Position suggérée: {pos:.0f} actions")
        
        if create_dashboard and modules.get('TradingDashboard'):
            print("\n📊 Création du dashboard...")
            try:
                overview = modules['get_stock_overview'](symbol)
                dashboard = modules['TradingDashboard'](symbol, results['current_price'])
                
                # Récupérer les données techniques
                technical_data = getattr(predictor, 'features', None)
                if technical_data is None:
                    print("⚠️ Aucune donnée technique disponible")
                
                # Construire le DataFrame de prédictions
                predictions_df = pd.DataFrame()
                if advanced:
                    # Pour mode avancé, appeler predict_future pour obtenir une série complète
                    future_results = predictor.predict_future(days_ahead=90)
                    predictions_df = future_results.get('predictions', pd.DataFrame())
                else:
                    # Mode simple : interpoler à partir des prédictions ponctuelles
                    last_date = predictor.data.index[-1] if predictor.data is not None else pd.Timestamp.now()
                    horizons_map = {'1d': 1, '5d': 5, '10d': 10, '20d': 20, '30d': 30, '90d': 90}
                    days = []
                    prices = []
                    for h_key, day in horizons_map.items():
                        if h_key in predictions and predictions[h_key] is not None:
                            days.append(day)
                            prices.append(predictions[h_key])
                    if len(days) >= 2:
                        all_days = np.arange(1, 91)
                        interpolated_prices = np.interp(all_days, days, prices)
                        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90, freq='B')
                        predictions_df = pd.DataFrame({'Predicted_Close': interpolated_prices}, index=future_dates)
                    elif prices:
                        future_dates = [last_date + timedelta(days=horizons_map[h_key]) for h_key in horizons_map if h_key in predictions]
                        predictions_df = pd.DataFrame({'Predicted_Close': prices}, index=future_dates)
                
                dashboard.load_data(
                    overview=overview,
                    technical_data=technical_data,
                    predictions_df=predictions_df,
                    score=results['trading_score'],
                    recommendation=results['recommendation'],
                    risk_metrics=results.get('risk_metrics', {}),
                    macro_data=results.get('market_context', {}),
                    market_sentiment=results.get('market_context', {})  # Assumer que le sentiment est inclus dans market_context
                )
                
                fig = dashboard.create_main_dashboard()
                if fig:
                    os.makedirs("dashboards", exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"dashboards/{symbol}_{timestamp}.html"
                    fig.write_html(filename)
                    print(f"✅ Dashboard sauvegardé: {filename}")
                    print("📄 Ouvrez ce fichier dans votre navigateur")
            except Exception as e:
                print(f"⚠️ Erreur création dashboard: {e}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n⏱️ Analyse terminée en {duration:.1f} secondes")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Système d'Analyse Boursière avec IA")
    parser.add_argument("symbol", nargs="?", help="Symbole boursier (ex: AAPL)")
    parser.add_argument("--period", default="1y", choices=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                        help="Période d'analyse")
    parser.add_argument("--interactive", action="store_true", help="Lancer le mode interactif (choix des actions)")
    parser.add_argument("--mode", choices=["analyze", "train", "dashboard", "compare", "screen"],
                        default="analyze", help="Mode d'exécution")
    parser.add_argument("--advanced", action="store_true", help="Utiliser le module avancé StockPredictor")
    parser.add_argument("--dashboard", action="store_true", help="Créer un dashboard après analyse")
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    print_banner()
    
    # Importer les modules
    print("\n📦 Chargement des modules...")
    modules = import_modules()
    
    if modules is None:
        print("❌ Impossible de charger les modules nécessaires")
        print(" Vérifiez que tous les fichiers sont correctement installés")
        return
    
    # Mode screening
    if args.mode == "screen":
        run_screening(modules)
        return
    
    # Mode comparaison
    if args.mode == "compare":
        if not args.symbol:
            print("❌ Veuillez spécifier des symboles à comparer (ex: AAPL,MSFT,GOOGL)")
            return
        
        symbols = [s.strip().upper() for s in args.symbol.split(',')]
        if len(symbols) < 2:
            print("❌ Veuillez spécifier au moins 2 symboles")
            return
        
        compare_stocks(symbols, args.period, modules)
        return
    
    # Mode analyse/train/dashboard pour une action
    if args.symbol:
        symbol = args.symbol.upper()
    else:
        # Mode interactif
        popular_stocks = {
            '1': 'AAPL',
            '2': 'TSLA',
            '3': 'MSFT',
            '4': 'GOOGL',
            '5': 'AMZN',
            '6': 'META',
            '7': 'NVDA',
            '8': 'SPY',
        }
        
        print("\n📈 ACTIONS POPULAIRES:")
        for key, value in popular_stocks.items():
            print(f" {key}. {value}")
        print(" 0. Entrer un symbole personnalisé")
        
        choice = input("\nChoisissez une action (1-8) ou 0 pour personnalisé: ").strip()
        
        if choice == '0':
            symbol = input("Entrez le symbole de l'action (ex: AAPL): ").strip().upper()
        elif choice in popular_stocks:
            symbol = popular_stocks[choice]
        else:
            print("Choix invalide, utilisation de AAPL par défaut")
            symbol = 'AAPL'
    
    # Exécuter l'analyse
    analyze_stock(
        symbol=symbol,
        period=args.period,
        mode=args.mode,
        advanced=args.advanced,
        create_dashboard=args.dashboard or args.mode == "dashboard",
        modules=modules
    )

if __name__ == "__main__":
    main()