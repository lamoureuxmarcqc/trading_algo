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
    from trading_algo.visualization.symbol_dashboard import AdvancedTradingDashboard
    try:
        dash = AdvancedTradingDashboard(symbol)

        # Prefer predictor-provided data (already fetched/trained)
        tech_df = getattr(predictor, "features", None)
        preds_df = None
        # predictions provided by results (mapping) -> convert to DataFrame if present
        preds_raw = results.get("predictions") if isinstance(results, dict) else None
        if isinstance(preds_raw, dict) and preds_raw:
            import pandas as pd
            try:
                preds_df = pd.DataFrame({"Predicted_Close": list(preds_raw.values())})
                start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
                preds_df.index = pd.date_range(start=start, periods=len(preds_df), freq="D")
            except Exception:
                preds_df = pd.DataFrame()

        # Only call SymbolManager if predictor has no useful technical data
        if (tech_df is None or (hasattr(tech_df, "empty") and tech_df.empty)):
            try:
                from trading_algo.stock.stock_manager import symbol_manager
                resp = symbol_manager.analyze_symbol(symbol) or {}
                tech_df = resp.get("technical") or tech_df
                if preds_df is None:
                    preds_df = resp.get("predictions") or preds_df
            except Exception:
                pass

        # Risk metrics preferences
        risk_metrics = results.get("risk_metrics", {}) if isinstance(results, dict) else {}

        # Load into dashboard (tolerant API)
        try:
            dash.load_data(
                technical_data=tech_df,
                predictions_df=preds_df,
                risk_metrics=risk_metrics,
                overview=results.get("overview", {})
            )
        except TypeError:
            # fallback to positional
            try:
                dash.load_data(tech_df, preds_df, risk_metrics)
            except Exception:
                dash.technical_data = tech_df
                dash.predictions_df = preds_df
                dash.risk_metrics = risk_metrics

        # propagate score/recommendation
        if hasattr(dash, "score"):
            dash.score = results.get("trading_score", getattr(predictor, "trading_score", dash.score))
        if hasattr(dash, "recommendation"):
            dash.recommendation = results.get("recommendation", getattr(predictor, "recommendation", dash.recommendation))

        # create figure (try consistent API)
        if hasattr(dash, "create_full_dashboard"):
            fig = dash.create_full_dashboard()
        elif hasattr(dash, "create_dashboard"):
            fig = dash.create_dashboard()
        else:
            creator = getattr(dash, "create", None)
            fig = creator() if callable(creator) else None

        if fig is None:
            logger.error("Erreur: impossible de créer la figure du dashboard.")
            return

        import os
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

# --- Mode Batch (Entraînement multi‑symboles) ---
def run_batch(args):
    """
    Lance l'entraînement batch sur une liste de symboles.
    """
    # Chargement dynamique du BatchTrainer
    BatchTrainer = get_module('trading_algo.batch.trainer', 'BatchTrainer')
    if not BatchTrainer:
        logger.error("❌ Impossible de charger BatchTrainer. Vérifiez que trading_algo.batch.trainer existe.")
        return

    # Déterminer la liste des symboles
    symbols = []
    if args.symbols_file:
        # Lecture depuis un fichier (un symbole par ligne)
        try:
            with open(args.symbols_file, 'r') as f:
                symbols = [line.strip().upper() for line in f if line.strip()]
            logger.info(f"📋 Chargement de {len(symbols)} symboles depuis {args.symbols_file}")
        except Exception as e:
            logger.error(f"Erreur lecture fichier symboles: {e}")
            return
    elif args.symbol:
        # Liste séparée par virgules
        symbols = [s.strip().upper() for s in args.symbol.split(',')]
        logger.info(f"📋 Utilisation des symboles: {symbols}")
    else:
        # Par défaut: un petit échantillon pour test
        default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"]
        # Ajout de quelques entreprises supplémentaires
        default_symbols.extend(["NFLX", "ADBE", "CRM", "INTC", "CSCO"])
        logger.warning(f"Aucun symbole spécifié. Utilisation des 10 valeurs par défaut: {default_symbols}")
        symbols = default_symbols

    if not symbols:
        logger.error("Aucun symbole à traiter.")
        return

    # Paramètres batch
    period = getattr(args, 'period', '10y')
    lookback = getattr(args, 'lookback', 60)
    epochs = getattr(args, 'batch_epochs', 50)
    batch_size = getattr(args, 'batch_size', 64)
    output_dir = getattr(args, 'output_dir', 'models_saved/batch')
    val_ratio = getattr(args, 'val_ratio', 0.1)
    clip_quantile = getattr(args, 'clip_quantile', 0.01)
    max_symbols = getattr(args, 'max_symbols', 500)
    min_rows = getattr(args, 'min_rows', 300)

    logger.info("=" * 60)
    logger.info("🚀 LANCEMENT DU MODE BATCH (entraînement multi‑symboles)")
    logger.info(f"Symboles: {len(symbols)} actions")
    logger.info(f"Période: {period}, Lookback: {lookback}, Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}, Validation ratio: {val_ratio}")
    logger.info(f"Dossier sortie: {output_dir}")
    logger.info("=" * 60)

    # Instanciation et exécution
    trainer = BatchTrainer(
        symbols=symbols,
        period=period,
        lookback=lookback,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        max_symbols=max_symbols,
        min_rows=min_rows,
        val_ratio=val_ratio,
        clip_quantile=clip_quantile,
        max_retries_per_symbol=3,
    )
    trainer.run()

# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Système d'Analyse Boursière avec IA")
    parser.add_argument("symbol", nargs="?", help="Symbole boursier (ou liste séparée par virgules en mode batch)")
    parser.add_argument("--mode", choices=["analyze", "train", "dashboard", "compare", "screen"], default="analyze")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--advanced", action="store_true")
    parser.add_argument("--portfolio", choices=["create", "load", "analyze", "rebalance"])
    parser.add_argument("--portfolio-name", default="default")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--web", action="store_true")
    
    # Arguments pour le mode batch
    parser.add_argument("--batch", action="store_true", help="Activer le mode batch (entraînement multi‑symboles)")
    parser.add_argument("--symbols-file", type=str, help="Fichier contenant une liste de symboles (un par ligne)")
    parser.add_argument("--lookback", type=int, default=60, help="Fenêtre rétrospective (jours)")
    parser.add_argument("--batch-epochs", type=int, default=50, help="Nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Taille du batch pour l'entraînement")
    parser.add_argument("--output-dir", type=str, default="models_saved/batch", help="Dossier de sauvegarde des modèles")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction des séquences pour validation")
    parser.add_argument("--clip-quantile", type=float, default=0.01, help="Quantile pour le clipping des outliers")
    parser.add_argument("--max-symbols", type=int, default=500, help="Nombre maximum de symboles à traiter")
    parser.add_argument("--min-rows", type=int, default=300, help="Nombre minimum de lignes alignées par symbole")
    
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
            run_batch(args)
        else:
            run_analysis(args)

    except KeyboardInterrupt:
        logger.info("\n👋 Interruption par l'utilisateur. Fermeture...")
    except Exception as e:
        logger.critical(f"💥 Erreur critique : {e}", exc_info=True)

if __name__ == "__main__":
    main()