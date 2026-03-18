#!/usr/bin/env python3
"""
Point d'entrée principal du système d'analyse boursière
"""
import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add near top of file, before other imports that rely on env
from pathlib import Path
try:
    from dotenv import load_dotenv
    # project root = parent of trading_algo package dir
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
except Exception:
    # python-dotenv not installed or no .env found: continue, rely on real env
    pass
# Centralized logging initialization for CLI entrypoint
from trading_algo.logging_config import init_logging

# Initialize logging once per process; logfile optional via env LOG_FILE
init_logging(level=os.getenv("LOG_LEVEL", None), logfile=os.getenv("LOG_FILE", None))
logger = logging.getLogger(__name__)


def print_banner():
    banner = r"""
╔══════════════════════════════════════════════════════════════╗
║ SYSTÈME D'ANALYSE BOURSIÈRE AVEC IA ║
║ 🚀 Trading Algorithmique Avancé 🚀 ║
╚══════════════════════════════════════════════════════════════╝
    """
    logger.info("\n" + banner)


def import_modules():
    """Importe tous les modules nécessaires"""
    modules = {}
    
    try:
        from trading_algo.data.data_extraction import StockDataExtractor, get_stock_overview, MacroDataExtractor
        modules['StockDataExtractor'] = StockDataExtractor
        modules['get_stock_overview'] = get_stock_overview
        modules['MacroDataExtractor'] = MacroDataExtractor
        logger.info("✅ Module data_extraction importé")
    except ImportError as e:
        logger.error(f"❌ Erreur data_extraction: {e}")
        return None
    
    try:
        from trading_algo.models.stockmodeltrain import StockModelTrain
        modules['StockModelTrain'] = StockModelTrain
        logger.info("✅ Module stockmodeltrain importé")
    except ImportError as e:
        logger.error(f"❌ Erreur stockmodeltrain: {e}")
        return None
    
    try:
        from trading_algo.models.stockpredictor import StockPredictor
        modules['StockPredictor'] = StockPredictor
        logger.info("✅ Module stockpredictor importé")
    except ImportError as e:
        logger.warning(f"⚠️ Module stockpredictor non disponible: {e}")
        modules['StockPredictor'] = None
    
    try:
        from trading_algo.visualization.dashboard import TradingDashboard, create_comparison_dashboard
        modules['TradingDashboard'] = TradingDashboard
        modules['create_comparison_dashboard'] = create_comparison_dashboard
        logger.info("✅ Module dashboard importé")
    except ImportError as e:
        logger.warning(f"⚠️ Module dashboard non disponible: {e}")
        modules['TradingDashboard'] = None
        modules['create_comparison_dashboard'] = None
    
    # Module screening optionnel
    try:
        from trading_algo.screening.actions_sp500 import screen_sp500, get_sp500_symbols
        modules['screen_sp500'] = screen_sp500
        modules['get_sp500_symbols'] = get_sp500_symbols
        logger.info("✅ Module actions_sp500 importé")
    except ImportError:
        modules['screen_sp500'] = None
        modules['get_sp500_symbols'] = None
    
    # Module portfolio
    try:
        from trading_algo.portfolio import Portfolio, Position, Order, PortfolioManager
        modules['Portfolio'] = Portfolio
        modules['Position'] = Position
        modules['Order'] = Order
        modules['PortfolioManager'] = PortfolioManager
        logger.info("✅ Module portfolio importé")
    except ImportError as e:
        logger.warning(f"⚠️ Module portfolio non disponible: {e}")
        for name in ['Portfolio', 'Position', 'Order', 'PortfolioManager']:
            modules[name] = None
    
    # Module portfoliodashboard (dashboard spécifique au portefeuille)
    try:
        from trading_algo.visualization.portfoliodashboard import PortfolioDashboard
        modules['PortfolioDashboard'] = PortfolioDashboard
        logger.info("✅ Module portfoliodashboard importé")
    except ImportError as e:
        logger.warning(f"⚠️ Module portfoliodashboard non disponible: {e}")
        modules['PortfolioDashboard'] = None
    
    return modules


def run_screening(modules):
    """Exécute le screening du S&P 500"""
    logger.info("\n🔍 SCREENING DU S&P 500")
    logger.info("=" * 60)
    
    screen_sp500 = modules.get('screen_sp500')
    if screen_sp500 is None:
        logger.error("❌ Module de screening non disponible")
        return
    
    try:
        results = screen_sp500(
            lookback_years=3,
            min_probability=0.65,
            max_stocks=15
        )
        
        if results['success']:
            stocks = results['stocks']
            
            logger.info("📊 RÉSULTATS:")
            logger.info(f" 🎯 Précision du modèle: {results['metrics']['accuracy']:.3f}")
            logger.info(f" 📈 Stocks recommandés: {len(stocks)}")
            
            if stocks:
                logger.info("🏆 ACTIONS RECOMMANDÉES:")
                for i, stock in enumerate(stocks[:10], 1):
                    logger.info(f" {i}. {stock['symbol']}:")
                    logger.info(f" Prix: ${stock['current_price']:.2f}")
                    logger.info(f" Probabilité: {stock['buy_probability']:.1%}")
                    logger.info(f" Signal: {stock['signal_strength']}")
                    if stock.get('rsi'):
                        rsi = stock['rsi']
                        status = "SURACHAT ⚠️" if rsi > 70 else "SURVENTE ✅" if rsi < 30 else "NEUTRE"
                        logger.info(f" RSI: {rsi:.1f} ({status})")
                    logger.info("")
            
            logger.info(f"💾 Résultats sauvegardés dans: {results['output_file']}")
            logger.info("💡 Analysez une action avec: trading-algo SYMBOLE --advanced")
        else:
            logger.error(f"❌ Erreur: {results['error']}")
            
    except Exception as e:
        logger.error(f"❌ Erreur screening: {e}", exc_info=True)
    
    logger.info("=" * 60)


def compare_stocks(symbols, period, modules):
    """Compare plusieurs actions"""
    logger.info(f"\n📊 COMPARAISON DE {len(symbols)} ACTIONS")
    logger.info("=" * 60)
    
    try:
        data_dict = {}
        extractor_class = modules['StockDataExtractor']
        
        for symbol in symbols:
            logger.info(f"\n📈 Récupération des données pour {symbol}...")
            extractor = extractor_class(symbol)
            data = extractor.get_historical_data(period=period)
            
            if not data.empty:
                data_dict[symbol] = data
                logger.info(f" ✅ {len(data)} périodes récupérées")
            else:
                logger.warning(f" ❌ Données non disponibles")
        
        if len(data_dict) > 1 and modules.get('create_comparison_dashboard'):
            logger.info("\n📊 Création du dashboard de comparaison...")
            fig = modules['create_comparison_dashboard'](list(data_dict.keys()), data_dict)
            
            if fig:
                os.makedirs("dashboards", exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"dashboards/comparison_{'_'.join(symbols)}_{timestamp}.html"
                fig.write_html(filename)
                logger.info(f"\n✅ Dashboard sauvegardé: {filename}")
                logger.info("📄 Ouvrez ce fichier dans votre navigateur")
        else:
            logger.warning("❌ Pas assez de données ou module dashboard non disponible")
            
    except Exception as e:
        logger.error(f"❌ Erreur comparaison: {e}", exc_info=True)


def analyze_stock(symbol, period, mode, advanced, create_dashboard, modules):
    """Analyse une action"""
    logger.info(f"\n🔍 Analyse pour {symbol} - période: {period}")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
        if advanced and modules.get('StockPredictor'):
            logger.info("🧠 Utilisation du module avancé StockPredictor...")
            predictor = modules['StockPredictor'](symbol, period)
        else:
            logger.info("🤖 Utilisation du module de base StockModelTrain...")
            predictor = modules['StockModelTrain'](symbol, period)
        
        if mode == "train":
            logger.info("\n🤖 Entraînement du modèle...")
            success = predictor.train(
                lookback_days=60,
                epochs=30
            )
            if success:
                logger.info("✅ Modèle entraîné avec succès!")
            else:
                logger.error("❌ Échec de l'entraînement")
            return
        
        logger.info(f"\n🔍 Analyse de {symbol}...")
        
        if advanced and modules.get('StockPredictor'):
            results = predictor.analyze_stock_advanced()
        else:
            results = predictor.analyze_model_stock()
        
        if 'error' in results:
            logger.error(f"❌ Erreur: {results['error']}")
            return
        
        logger.info(f"\n📊 RÉSULTATS POUR {symbol}:")
        logger.info(f" 📈 Prix actuel: ${results['current_price']:.2f}")
        logger.info(f" 🎯 Score de trading: {results['trading_score']}/10")
        logger.info(f" 💡 Recommandation: {results['recommendation']}")
        
        predictions = results.get('predictions', {})
        if predictions:
            logger.info("\n 🔮 Prédictions de prix:")
            for horizon in ['1d', '5d', '10d', '20d', '30d', '90d']:
                if horizon in predictions and predictions[horizon] is not None:
                    pred = predictions[horizon]
                    change_pct = ((pred - results['current_price']) / results['current_price'] * 100)
                    logger.info(f" {horizon}: ${pred:.2f} ({change_pct:+.2f}%)")
        
        # Affichage des métriques de risque si disponibles
        if 'risk_metrics' in results and advanced:
            rm = results['risk_metrics']
            logger.info("\n⚠️ MÉTRIQUES DE RISQUE:")
            if 'sharpe_ratio' in rm:
                logger.info(f" Sharpe Ratio: {rm['sharpe_ratio']:.2f}")
            if 'max_drawdown' in rm:
                logger.info(f" Max Drawdown: {rm['max_drawdown'] * 100:.1f}%")
            if 'value_at_risk' in rm:
                logger.info(f" Value at Risk (95%): {rm['value_at_risk'] * 100:.1f}%")
            if 'beta' in rm:
                logger.info(f" Bêta: {rm['beta']:.2f}")
            
            if 'stop_loss_levels' in rm:
                logger.info("\n Niveaux de gestion du risque:")
                for horizon in ['1d', '5d', '10d', '20d', '30d', '90d']:
                    if horizon in rm['stop_loss_levels']:
                        stop = rm['stop_loss_levels'][horizon]
                        tp = rm['take_profit_levels'].get(horizon, 'N/A')
                        rr = rm['risk_reward_ratios'].get(horizon, 'N/A')
                        pos = rm['suggested_position_sizes'].get(horizon, 'N/A')
                        logger.info(f" {horizon}: Stop ${stop:.2f}, Target ${tp:.2f}, R/R {rr:.2f}, Position suggérée: {pos:.0f} actions")
        
        if create_dashboard and modules.get('TradingDashboard'):
            logger.info("\n📊 Création du dashboard...")
            try:
                overview = modules['get_stock_overview'](symbol)
                dashboard = modules['TradingDashboard'](symbol, results['current_price'])
                
                # Récupérer les données techniques
                technical_data = getattr(predictor, 'features', None)
                if technical_data is None:
                    logger.warning("⚠️ Aucune donnée technique disponible")
                
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
                    logger.info(f"✅ Dashboard sauvegardé: {filename}")
                    logger.info("📄 Ouvrez ce fichier dans votre navigateur")
            except Exception as e:
                logger.warning(f"⚠️ Erreur création dashboard: {e}", exc_info=True)
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\n⏱️ Analyse terminée en {duration:.1f} secondes")
    logger.info("=" * 60)


def manage_portfolio(args, modules):
    """Gère les opérations de portefeuille"""
    # Récupération des classes depuis modules
    Portfolio = modules['Portfolio']
    PortfolioManager = modules['PortfolioManager']
    PortfolioDashboard = modules.get('PortfolioDashboard')  # Optionnel
    
    if Portfolio is None or PortfolioManager is None:
        logger.error("❌ Module portfolio non disponible")
        return
    
    # Créer le gestionnaire
    manager = PortfolioManager(modules['StockDataExtractor'])
    
    if args.portfolio == "create":
        if not args.initial_cash:
            logger.error("❌ Spécifiez --initial-cash pour créer un portefeuille")
            return
        
        portfolio = manager.create_portfolio(args.portfolio_name, args.initial_cash)
        manager.save_current_portfolio()
        logger.info(f"✅ Portefeuille '{args.portfolio_name}' créé avec {args.initial_cash}$")
    
    elif args.portfolio == "load":
        portfolio = manager.load_portfolio(args.portfolio_name)
        if portfolio:
            logger.info(f"✅ Portefeuille '{args.portfolio_name}' chargé")
            logger.info(f"💰 Cash: {portfolio.cash:.2f}$")
            logger.info(f"📊 Positions: {len(portfolio.positions)}")
        else:
            logger.error(f"❌ Portefeuille '{args.portfolio_name}' non trouvé")
    
    elif args.portfolio == "analyze":
        portfolio = manager.load_portfolio(args.portfolio_name)
        if not portfolio:
            logger.error(f"❌ Portefeuille '{args.portfolio_name}' non trouvé")
            return
        
        logger.info(f"\n📊 ANALYSE DU PORTEFEUILLE: {portfolio.name}")
        logger.info("=" * 60)
        
        analysis = manager.analyze_portfolio()
        
        perf = analysis['performance']
        logger.info("\n💰 PERFORMANCE:")
        logger.info(f" Valeur totale: ${perf['total_value']:.2f}")
        logger.info(f" Liquidités: ${perf['cash']:.2f}")
        logger.info(f" Investi: ${perf['invested']:.2f}")
        logger.info(f" P&L total: ${perf['total_pnl']:.2f} ({perf['total_pnl_pct']:.2f}%)")
        
        logger.info("\n📈 ALLOCATION:")
        for ticker, alloc in analysis['allocation'].items():
            if alloc > 0.01:
                logger.info(f" {ticker}: {alloc*100:.1f}%")
        
        if args.dashboard and PortfolioDashboard is not None:
            # Créer le dashboard
            dashboard = PortfolioDashboard(portfolio, manager)
            fig = dashboard.create_portfolio_dashboard(analysis['market_prices'])
            
            if fig:
                os.makedirs("dashboards", exist_ok=True)
                filename = f"dashboards/portfolio_{args.portfolio_name}.html"
                fig.write_html(filename)
                logger.info(f"\n✅ Dashboard sauvegardé: {filename}")
        elif args.dashboard and PortfolioDashboard is None:
            logger.warning("⚠️ PortfolioDashboard non disponible, impossible de créer le dashboard")
    
    elif args.portfolio == "rebalance":
        if not args.symbol:
            logger.error("❌ Spécifiez les allocations cibles (ex: AAPL:0.4,MSFT:0.3,cash:0.3)")
            return
        
        portfolio = manager.load_portfolio(args.portfolio_name)
        if not portfolio:
            logger.error(f"❌ Portefeuille '{args.portfolio_name}' non trouvé")
            return
        
        # Parser les allocations
        target_allocation = {}
        for item in args.symbol.split(','):
            ticker, pct = item.split(':')
            target_allocation[ticker] = float(pct)
        
        orders, impact = manager.suggest_rebalance(target_allocation)
        
        logger.info(f"\n🔄 RÉÉQUILIBRAGE PROPOSÉ")
        logger.info("=" * 60)
        logger.info(f" Ordres: {impact['orders_count']}")
        logger.info(f" Volume total: ${impact['total_trade_value']:.2f}")
        logger.info(f" Impact: {impact['total_trade_pct']*100:.1f}% du portefeuille")
        
        if orders:
            logger.info(f"\n📝 ORDRES À EXÉCUTER:")
            for order in orders:
                logger.info(f" {order.order_type.upper()} {order.quantity:.2f} {order.ticker} @ ${order.limit_price:.2f}")
            
            confirm = input("\nExécuter ces ordres? (o/n): ").lower()
            if confirm == 'o':
                # Exécuter les ordres
                market_prices = manager.get_market_prices(list(target_allocation.keys()))
                for order in orders:
                    if order.order_type == 'buy':
                        portfolio.add_position(order.ticker, order.quantity, market_prices[order.ticker])
                    else:
                        portfolio.remove_position(order.ticker, order.quantity, market_prices[order.ticker])
                
                manager.save_current_portfolio()
                logger.info("✅ Ordres exécutés et portefeuille sauvegardé")


def main():
    parser = argparse.ArgumentParser(description="Système d'Analyse Boursière avec IA")
    parser.add_argument("symbol", nargs="?", help="Symbole boursier (ex: AAPL)")
    parser.add_argument("--period", default="3y", choices=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                        help="Période d'analyse")
    parser.add_argument("--interactive", action="store_true", help="Lancer le mode interactif (choix des actions)")
    parser.add_argument("--mode", choices=["analyze", "train", "dashboard", "compare", "screen"],
                        default="analyze", help="Mode d'exécution")
    parser.add_argument("--advanced", action="store_true", help="Utiliser le module avancé StockPredictor")
    parser.add_argument("--portfolio", help="Gérer un portefeuille (create, load, analyze, rebalance)")
    parser.add_argument("--portfolio-name", default="default", help="Nom du portefeuille")
    parser.add_argument("--initial-cash", type=float, help="Cash initial pour créer un portefeuille")
    parser.add_argument("--dashboard", action="store_true", help="Créer un dashboard après analyse")
    parser.add_argument("--web", action="store_true", help="Lancer l'interface web Dash")
    parser.add_argument("--batch", action="store_true", help="Lancer le mode batch d'entraînement (unified model)")
    parser.add_argument("--batch-symbols-file", help="Fichier (une ligne par ticker) pour le mode batch")
    parser.add_argument("--batch-epochs", type=int, default=30, help="Epochs pour le mode batch")
    parser.add_argument("--batch-lookback", type=int, default=60, help="Lookback jours pour le mode batch")
    parser.add_argument("--batch-max-symbols", type=int, default=500, help="Nombre maximum de symboles pour batch")


    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    print_banner()
    
    # Importer les modules
    logger.info("\n📦 Chargement des modules...")
    modules = import_modules()
    
    if modules is None:
        logger.error("❌ Impossible de charger les modules nécessaires")
        logger.error(" Vérifiez que tous les fichiers sont correctement installés")
        return
    
    # Dans la fonction main(), après le chargement des modules
    if args.web:
        logger.info("🌐 Lancement de l'interface web Dash...")
        try:
            from trading_algo.web_dashboard.app import app
            app.run(debug=True, host='localhost', port=8050)
        except Exception as e:
            logger.error(f"❌ Erreur lors du lancement du serveur web: {e}", exc_info=True)
        return    
    
    # Mode portfolio
    if args.portfolio:
        manage_portfolio(args, modules)
        return
    
    # Mode screening
    if args.mode == "screen":
        run_screening(modules)
        return
    
    # Mode comparaison
    if args.mode == "compare":
        if not args.symbol:
            logger.error("❌ Veuillez spécifier des symboles à comparer (ex: AAPL,MSFT,GOOGL)")
            return
        
        symbols = [s.strip().upper() for s in args.symbol.split(',')]
        if len(symbols) < 2:
            logger.error("❌ Veuillez spécifier au moins 2 symboles")
            return
        
        compare_stocks(symbols, args.period, modules)
        return

    # Mode batch: entraînement d'un modèle unique sur plusieurs symboles
    if args.batch:
        logger.info("🔁 Lancement du mode batch d'entraînement")
        try:
            from trading_algo.batch.trainer import BatchTrainer
            # load symbols from file if provided, else try S&P500 list
            symbols = []
            if args.batch_symbols_file:
                with open(args.batch_symbols_file, 'r', encoding='utf-8') as f:
                    symbols = [line.strip().upper() for line in f if line.strip()]
            elif modules.get('get_sp500_symbols'):
                symbols = modules['get_sp500_symbols']()
            else:
                # fallback minimal set
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

            trainer = BatchTrainer(
                symbols=symbols,
                period="20y",
                lookback=args.batch_lookback,
                epochs=args.batch_epochs,
                batch_size=32,
                output_dir="models_saved/batch",
                max_symbols=args.batch_max_symbols,
            )
            trainer.run()
        except Exception as e:
            logger.error(f"Erreur mode batch: {e}", exc_info=True)
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
        
        logger.info("\n📈 ACTIONS POPULAIRES:")
        for key, value in popular_stocks.items():
            logger.info(f" {key}. {value}")
        logger.info(" 0. Entrer un symbole personnalisé")
        
        choice = input("\nChoisissez une action (1-8) ou 0 pour personnalisé: ").strip()
        
        if choice == '0':
            symbol = input("Entrez le symbole de l'action (ex: AAPL): ").strip().upper()
        elif choice in popular_stocks:
            symbol = popular_stocks[choice]
        else:
            logger.warning("Choix invalide, utilisation de AAPL par défaut")
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
    # Test rapide du module portfolio (optionnel)
    '''try:
        from trading_algo.portfolio import Portfolio
        p = Portfolio(cash=10000)
        p.add_position('AAPL', 10, 150)
        logger.info("✅ Test portfolio réussi")
    except Exception as e:
        logger.error(f"❌ Test portfolio échoué: {e}", exc_info=True)
     '''
    main()
