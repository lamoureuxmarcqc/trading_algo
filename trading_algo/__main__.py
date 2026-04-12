#!/usr/bin/env python3
"""
Système d'Analyse Boursière avec IA - Point d'entrée optimisé
"""
import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import trading_algo.settings


# --- Configuration Initiale ---
try:
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
except Exception:
    pass

from trading_algo.logging_config import init_logging
init_logging(level=os.getenv("LOG_LEVEL", "INFO"), logfile=os.getenv("LOG_FILE", None))
logger = logging.getLogger(__name__)

def print_banner():
    banner = r"""
╔══════════════════════════════════════════════════════════════╗
║        SYSTÈME D'ANALYSE BOURSIÈRE AVEC IA           ║
║         🚀 Trading Algorithmique Avancé 🚀           ║
╚══════════════════════════════════════════════════════════════╝"""
    print(banner)

# --- Utilitaires de Chargement (Lazy Loading) ---
def get_module(path, name):
    import importlib
    try:
        mod = importlib.import_module(path)
        return getattr(mod, name)
    except (ImportError, AttributeError) as e:
        logger.error(f"❌ Impossible de charger {name} depuis {path}: {e}")
        return None

# --- Fonctions de Mode ---

def run_screening(args):
    screen_sp500 = get_module('trading_algo.screening.actions_sp500', 'screen_sp500')
    if not screen_sp500: return
    
    logger.info("\n🔍 Lancement du screening S&P 500...")
    results = screen_sp500(lookback_years=3, min_probability=0.65, max_stocks=15)
    
    if results.get('success'):
        for i, stock in enumerate(results['stocks'][:10], 1):
            logger.info(f"{i}. {stock['symbol']} - Prob: {stock['buy_probability']:.1%} - RSI: {stock.get('rsi', 'N/A')}")
    else:
        logger.error(f"Erreur: {results.get('error')}")

def run_comparison(args):
    if not args.symbol:
        logger.error("❌ Spécifiez des symboles (ex: AAPL,MSFT)")
        return
    
    symbols = [s.strip().upper() for s in args.symbol.split(',')]
    Extractor = get_module('trading_algo.data.data_extraction', 'StockDataExtractor')
    CompareDash = get_module('trading_algo.visualization.symbol_dashboard', 'create_comparison_dashboard')
    
    def fetch(s):
        logger.info(f"📥 Récupération {s}...")
        data = Extractor(s).get_historical_data(period=args.period)
        return s, data

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = dict(executor.map(fetch, symbols))
    
    valid_data = {s: d for s, d in results.items() if not d.empty}
    if len(valid_data) > 1 and CompareDash:
        fig = CompareDash(list(valid_data.keys()), valid_data)
        path = f"dashboards/comp_{datetime.now().strftime('%H%M%S')}.html"
        os.makedirs("dashboards", exist_ok=True)
        fig.write_html(path)
        logger.info(f"✅ Comparaison sauvegardée: {path}")

def run_analysis(args):
    symbol = args.symbol.upper() if args.symbol else "AAPL"
    
    # Choix du modèle
    if args.advanced:
        PredictorClass = get_module('trading_algo.models.stockpredictor', 'StockPredictor')
    else:
        PredictorClass = get_module('trading_algo.models.stockmodeltrain', 'StockModelTrain')
    
    if not PredictorClass: return

    predictor = PredictorClass(symbol, args.period)
    
    if args.mode == "train":
        predictor.train(lookback_days=60, epochs=args.batch_epochs)
        return

    results = predictor.analyze_stock_advanced() if args.advanced else predictor.analyze_model_stock()
    
    if 'error' in results:
        logger.error(f"Erreur: {results['error']}")
        return

    logger.info(f"\n📊 {symbol} | Prix: ${results['current_price']:.2f} | Score: {results['trading_score']}/10")
    logger.info(f"💡 Rec: {results['recommendation']}")

    # Use the correct CLI flag attribute: args.dashboard
    if (args.dashboard or args.mode == "dashboard"):
        create_stock_dashboard(symbol, predictor, results)

def create_stock_dashboard(symbol, predictor, results):
    """
    Build stock dashboard using SymbolManager as primary data source and
    AdvancedTradingDashboard as the visualization layer.

    - Uses SymbolManager.analyze_symbol(symbol) to get normalized structures:
      {'results': ..., 'technical': pd.DataFrame, 'predictions': pd.DataFrame}
    - Falls back to predictor attributes when manager doesn't provide data.
    - Supports multiple AdvancedTradingDashboard API shapes (create_full_dashboard / create_dashboard).
    """
    from trading_algo.visualization.symbol_dashboard import AdvancedTradingDashboard
    from trading_algo.stock.stock_manager import symbol_manager
    import pandas as pd
    import os

    try:
        # Instantiate dashboard (single-arg constructor expected)
        dash = AdvancedTradingDashboard(symbol)

        # Prefer business-layer data from SymbolManager (lazy, robust)
        resp = {}
        try:
            resp = symbol_manager.analyze_symbol(symbol) or {}
        except Exception:
            resp = {}

        tech_df = resp.get("technical") if isinstance(resp, dict) else None
        preds_df = resp.get("predictions") if isinstance(resp, dict) else None
        mgr_results = resp.get("results", {}) if isinstance(resp, dict) else {}

        # Fallback to predictor-provided data if manager didn't return technicals
        if (tech_df is None or (hasattr(tech_df, "empty") and tech_df.empty)) and hasattr(predictor, "features"):
            tech_df = getattr(predictor, "features", tech_df)

        # Build a simple predictions DataFrame if none provided
        if preds_df is None:
            preds_raw = results.get("predictions") or mgr_results.get("predictions") or {}
            if isinstance(preds_raw, dict) and preds_raw:
                try:
                    # Convert mapping {'1d': price, ...} to a dated DataFrame (simple incremental dates)
                    preds_df = pd.DataFrame({"Predicted_Close": list(preds_raw.values())})
                    start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
                    preds_df.index = pd.date_range(start=start, periods=len(preds_df), freq="D")
                except Exception:
                    preds_df = pd.DataFrame()
            else:
                preds_df = pd.DataFrame()

        # Determine risk metrics: prefer manager's, fallback to results
        risk_metrics = mgr_results.get("risk_metrics", results.get("risk_metrics", {}))

        # Load data into the visualization layer using a tolerant call
        try:
            # Try the common signature first
            dash.load_data(
                technical_data=tech_df,
                predictions_df=preds_df,
                risk_metrics=risk_metrics
            )
        except TypeError:
            # Fallback: try positional or alternative names if load_data differs
            try:
                dash.load_data(tech_df, preds_df, risk_metrics)
            except Exception:
                # Last resort: set attributes directly
                setattr(dash, "technical_data", tech_df)
                setattr(dash, "predictions_df", preds_df)
                setattr(dash, "risk_metrics", risk_metrics)

        # Propagate score & recommendation if supported by dashboard
        if hasattr(dash, "score"):
            dash.score = results.get("trading_score", mgr_results.get("trading_score", getattr(predictor, "trading_score", 5.0)))
        if hasattr(dash, "recommendation"):
            dash.recommendation = results.get("recommendation", mgr_results.get("recommendation", getattr(predictor, "recommendation", "NEUTRE")))

        # Create figure using available API
        fig = None
        if hasattr(dash, "create_full_dashboard"):
            fig = dash.create_full_dashboard()
        elif hasattr(dash, "create_dashboard"):
            fig = dash.create_dashboard()
        else:
            # Try generic name
            creator = getattr(dash, "create", None)
            fig = creator() if callable(creator) else None

        if fig is None:
            logger.error("Erreur lors de la génération du dashboard visuel : figure introuvable.")
            return

        path = f"dashboards/{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        os.makedirs("dashboards", exist_ok=True)
        fig.write_html(path)
        logger.info(f"✅ Dashboard généré: {path}")

    except Exception as e:
        logger.error(f"Erreur Dashboard: {e}", exc_info=True)

def run_portfolio(args):
    ManagerClass = get_module('trading_algo.portfolio', 'PortfolioManager')
    ExtractorClass = get_module('trading_algo.data.data_extraction', 'StockDataExtractor')
    if not ManagerClass: return

    manager = ManagerClass(ExtractorClass)
    # Note: Implémenter ici la logique switch/case pour create, load, analyze, rebalance
    logger.info(f"Action portfolio: {args.portfolio} sur {args.portfolio_name}")

def run_web(args):
    app = get_module('trading_algo.web_dashboard.app', 'app')
    if app:
        logger.info("🌐 Démarrage du serveur Dash sur http://localhost:8050")
        app.run(debug=True, host='localhost', port=8050)

# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Système d'Analyse Boursière avec IA")
    parser.add_argument("symbol", nargs="?", help="Symbole boursier")
    parser.add_argument("--mode", choices=["analyze", "train", "dashboard", "compare", "screen"], default="analyze")
    parser.add_argument("--period", default="3y")
    parser.add_argument("--advanced", action="store_true")
    parser.add_argument("--portfolio", choices=["create", "load", "analyze", "rebalance"])
    parser.add_argument("--portfolio-name", default="default")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--batch-epochs", type=int, default=30)
    
    args = parser.parse_args()
    print_banner()

    try:
        if args.web:
            run_web(args)
        elif args.portfolio:
            run_portfolio(args)
        elif args.mode == "screen":
            run_screening(args)
        elif args.mode == "compare":
            run_comparison(args)
        elif args.batch:
            # Charger BatchTrainer via lazy loading ici si besoin
            logger.info("Mode batch non implémenté dans cet exemple.")
        else:
            run_analysis(args)

    except KeyboardInterrupt:
        logger.info("\n👋 Interruption par l'utilisateur. Fermeture...")
    except Exception as e:
        logger.critical(f"💥 Erreur critique : {e}", exc_info=True)

if __name__ == "__main__":
    main()