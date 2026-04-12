import pandas as pd
import logging
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class StockManager:
    """
    Module de reporting et visualisation (figures kept in visualization).
    Prend en entrée les données déjà calculées par DataExtractor.
    """
    def __init__(self, symbol: str, features: pd.DataFrame, targets: pd.DataFrame = None):
        self.symbol = symbol
        self.df = features  # Contient déjà RSI, MACD, ATR, etc.
        self.targets = targets

    def create_dashboard(self):
        # kept for backward compatibility; prefer visualization.AdvancedTradingDashboard
        from trading_algo.visualization.symbol_dashboard import AdvancedTradingDashboard
        td = AdvancedTradingDashboard(self.symbol)
        td.load_data(technical_data=self.df)
        return td.create_full_dashboard()

    def export_report(self, output_format="html", folder="reports"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig = self.create_dashboard()
        filename = f"{folder}/{self.symbol}_report.{output_format}"
        if output_format.lower() == "html":
            fig.write_html(filename)
        else:
            fig.write_image(filename, engine="kaleido")
        logger.info(f"Rapport {output_format} généré : {filename}")
        return filename


class SymbolManager:
    """
    Manager responsable de l'orchestration de l'analyse d'un symbole.
    Centralise l'accès à StockPredictor et renvoie des structures de données
    simples (dict + DataFrame) que la couche UI peut consommer.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_symbol(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """
        Lance l'analyse avancée pour un symbole et retourne:
        {
          'results': dict (issue par StockPredictor) OR {'error': '...'},
          'technical': pd.DataFrame or None,
          'predictions': pd.DataFrame or None
        }
        """
        try:
            from trading_algo.models.stockpredictor import StockPredictor
        except Exception as e:
            self.logger.exception("Failed to import StockPredictor: %s", e)
            return {'error': f"Import error: {e}"}

        try:
            predictor = StockPredictor(symbol, period)
            results = predictor.analyze_stock_advanced()
            if not results or 'error' in results:
                return results if isinstance(results, dict) else {'error': 'Empty analysis result'}

            tech_df = None
            for key in ('technical', 'features', 'historical', 'technical_indicators'):
                data = results.get(key)
                if isinstance(data, pd.DataFrame) and not data.empty:
                    tech_df = data
                    break

            if tech_df is None and hasattr(predictor, 'data') and isinstance(predictor.data, pd.DataFrame):
                if not predictor.data.empty:
                    tech_df = predictor.data.copy()

            pred_df = None
            preds = results.get('predictions')
            if isinstance(preds, dict) and preds:
                try:
                    pred_df = pd.DataFrame(preds)
                    if 'date' in pred_df.columns:
                        pred_df['date'] = pd.to_datetime(pred_df['date'])
                        pred_df.set_index('date', inplace=True)
                except Exception:
                    pred_df = None

            return {
                'results': results,
                'technical': tech_df,
                'predictions': pred_df
            }
        except Exception as e:
            self.logger.exception("Error while analyzing symbol %s: %s", symbol, e)
            return {'error': str(e)}


# singleton instance for easy import
symbol_manager = SymbolManager()
