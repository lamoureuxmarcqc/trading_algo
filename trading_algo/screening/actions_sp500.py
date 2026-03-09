"""
Module de screening des actions du S&P 500
Analyse et recommande des actions basées sur l'apprentissage automatique
"""

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Logging must be configured centrally by the entrypoint; modules should only fetch a logger
logger = logging.getLogger(__name__)


class StockScreener:
    """
    Classe pour le screening d'actions du S&P 500
    """
    
    def __init__(self, 
                 lookback_years: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialise le screener
        
        Args:
            lookback_years: Nombre d'années de données historiques
            test_size: Proportion des données pour le test
            random_state: Seed pour la reproductibilité
        """
        self.lookback_years = lookback_years
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.labels = None
        self.accuracy = None
        self.screened_stocks = []
        
        logger.info(f"StockScreener initialisé (lookback: {lookback_years} ans)")
    
    @staticmethod
    def get_sp500_symbols() -> List[str]:
        """
        Récupère la liste des symboles du S&P 500 depuis un fichier CSV local
        situé dans trading_algo/data/cie_sp500.csv.
        Si le fichier n'est pas trouvé, utilise une liste de secours.
        """
        # Chemin absolu vers le fichier CSV (en partant de l'emplacement de ce script)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Correction : utiliser os.path.join avec des composants séparés pour compatibilité multi-plateforme
        csv_path = os.path.join(current_dir, '..', 'data', 'cie_sp500.csv')
    
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Fichier CSV non trouvé: {csv_path}")
        
            df = pd.read_csv(csv_path)
        
            # Déterminer la colonne des symboles : on cherche 'Symbol' ou on prend la première colonne
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].tolist()
            else:
                symbols = df.iloc[:, 0].tolist()  # première colonne
        
            # Nettoyer les symboles : supprimer les espaces, remplacer '.' par '-' (compatible yfinance)
            symbols = [str(s).strip().replace('.', '-') for s in symbols]
        
            logger.info(f"Récupéré {len(symbols)} symboles depuis le fichier local")
            return symbols
        
        except Exception as e:
            logger.error(f"Erreur lecture du fichier CSV: {e}")
            # Liste de secours
            fallback = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JPM', 'V']
            logger.info(f"Utilisation de la liste de secours ({len(fallback)} symboles)")
            return fallback   
    def fetch_stock_data(self, 
                         symbols: List[str] = None,
                         max_symbols: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Récupère les données historiques pour les symboles
        
        Args:
            symbols: Liste des symboles à analyser
            max_symbols: Nombre maximum de symboles à traiter
        """
        if symbols is None:
            symbols = self.get_sp500_symbols()
        
        # Limiter le nombre de symboles pour éviter les timeouts
        symbols = symbols[:max_symbols]
        
        data = {}
        period = f"{self.lookback_years}y"
        
        logger.info(f"Récupération des données pour {len(symbols)} symboles...")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                if i % 10 == 0:
                    logger.info(f"  Progression: {i}/{len(symbols)}")
                
                stock = yf.Ticker(symbol)
                history = stock.history(period=period, interval='1d')
                
                if not history.empty and len(history) > 200:  # Au moins 200 jours de données
                    data[symbol] = history
                    
            except Exception as e:
                logger.warning(f"Erreur pour {symbol}: {e}")
                continue
        
        logger.info(f"Données récupérées pour {len(data)} symboles")
        return data
    
    def calculate_features(self, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Calcule les features pour l'entraînement du modèle
        """
        features_list = []
        labels_list = []
        
        for symbol, df in data.items():
            try:
                if len(df) < 50:  # Pas assez de données pour calculer SMA50
                    continue
                
                # Features de base
                df = df.copy()
                df['Returns'] = df['Close'].pct_change()
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['SMA_200'] = df['Close'].rolling(window=200).mean()
                df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
                
                # Indicateurs avancés
                df['RSI'] = self._calculate_rsi(df['Close'])
                df['MACD'] = self._calculate_macd(df['Close'])
                df['Volatility'] = df['Returns'].rolling(window=20).std()
                df['Momentum'] = df['Close'] / df['Close'].shift(20) - 1
                
                # Target: Le prix montera-t-il demain ? (1 = oui, 0 = non)
                df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
                
                # Supprimer les NaN
                df_clean = df.dropna()
                
                if len(df_clean) > 0:
                    # Features sélectionnées
                    feature_cols = [
                        'SMA_20', 'SMA_50', 'SMA_200',
                        'Volume', 'Volume_SMA',
                        'RSI', 'MACD', 'Volatility', 'Momentum'
                    ]
                    
                    # Vérifier que toutes les colonnes existent
                    available_cols = [col for col in feature_cols if col in df_clean.columns]
                    
                    if available_cols:
                        features_list.append(df_clean[available_cols])
                        labels_list.append(df_clean['Target'])
                        
            except Exception as e:
                logger.warning(f"Erreur calcul features pour {symbol}: {e}")
                continue
        
        if not features_list:
            raise ValueError("Aucune feature calculée")
        
        # Concaténer toutes les données
        features = pd.concat(features_list)
        labels = pd.concat(labels_list)
        
        # Vérifier l'alignement
        if len(features) != len(labels):
            raise ValueError(f"Incohérence features/labels: {len(features)} vs {len(labels)}")
        
        self.features = features
        self.labels = labels
        
        logger.info(f"Features calculées: {features.shape}")
        return features, labels
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calcule le MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        return macd
    
    def train_model(self, 
                   model_type: str = 'random_forest',
                   n_estimators: int = 100) -> Dict[str, Any]:
        """
        Entraîne le modèle de classification
        
        Args:
            model_type: Type de modèle ('random_forest', 'gradient_boosting')
            n_estimators: Nombre d'estimateurs pour les modèles d'ensemble
        """
        if self.features is None or self.labels is None:
            raise ValueError("Features ou labels non disponibles. Exécutez calculate_features d'abord.")
        
        logger.info(f"Entraînement du modèle {model_type}...")
        
        # Sélection du modèle
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Type de modèle inconnu: {model_type}")
        
        # Pipeline avec scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=False  # Garder l'ordre temporel
        )
        
        # Entraînement
        pipeline.fit(X_train, y_train)
        
        # Évaluation
        y_pred = pipeline.predict(X_test)
        
        # Métriques
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'model_type': model_type,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Validation croisée
        cv_scores = cross_val_score(pipeline, self.features, self.labels, cv=5)
        metrics['cv_mean_accuracy'] = cv_scores.mean()
        metrics['cv_std_accuracy'] = cv_scores.std()
        
        self.model = pipeline
        self.accuracy = metrics['accuracy']
        
        logger.info(f"Modèle entraîné - Accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"Précision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
        
        return metrics
    
    def screen_stocks(self, 
                     data: Dict[str, pd.DataFrame],
                     min_probability: float = 0.6,
                     max_stocks: int = 20) -> List[Dict[str, Any]]:
        """
        Screen les stocks basés sur le modèle entraîné
        
        Args:
            data: Dictionnaire des données historiques
            min_probability: Probabilité minimale pour recommander l'achat
            max_stocks: Nombre maximum de stocks à recommander
        """
        if self.model is None:
            raise ValueError("Modèle non entraîné. Exécutez train_model d'abord.")
        
        logger.info(f"Screening des stocks (min_probability: {min_probability})...")
        
        recommended_stocks = []
        
        for symbol, df in data.items():
            try:
                # Calculer les features pour la dernière ligne
                last_row = self._prepare_stock_features(df)
                
                if last_row is not None and len(last_row) > 0:
                    # Prédiction
                    proba = self.model.predict_proba([last_row])[0]
                    buy_probability = proba[1] # Probabilité de classe positive (achat)
                    
                    # Informations sur le stock
                    current_price = df['Close'].iloc[-1]
                    volume = df['Volume'].iloc[-1]
                    sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
                    
                    # RSI
                    rsi = self._calculate_rsi(df['Close']).iloc[-1] if len(df) >= 15 else None
                    
                    # Recommander si probabilité suffisante
                    if buy_probability >= min_probability:
                        stock_info = {
                            'symbol': symbol,
                            'current_price': float(current_price),
                            'buy_probability': float(buy_probability),
                            'volume': int(volume),
                            'sma_50': float(sma_50) if sma_50 else None,
                            'rsi': float(rsi) if rsi else None,
                            'price_vs_sma50': float((current_price - sma_50) / sma_50 * 100) if sma_50 else None,
                            'signal_strength': 'FORT' if buy_probability >= 0.8 else 'MOYEN' if buy_probability >= 0.6 else 'FAIBLE'
                        }
                        recommended_stocks.append(stock_info)
                        
            except Exception as e:
                logger.warning(f"Erreur screening {symbol}: {e}")
                continue
        
        # Trier par probabilité décroissante
        recommended_stocks.sort(key=lambda x: x['buy_probability'], reverse=True)
        
        # Limiter le nombre de résultats
        recommended_stocks = recommended_stocks[:max_stocks]
        
        self.screened_stocks = recommended_stocks
        
        logger.info(f"{len(recommended_stocks)} stocks recommandés pour achat")
        return recommended_stocks
    
    def _prepare_stock_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Prépare les features d'un stock pour la prédiction"""
        try:
            if len(df) < 50:
                return None
            
            df = df.copy()
            
            # Calculer les mêmes features que pendant l'entraînement
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['RSI'] = self._calculate_rsi(df['Close'])
            df['MACD'] = self._calculate_macd(df['Close'])
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            df['Momentum'] = df['Close'] / df['Close'].shift(20) - 1
            
            # Prendre la dernière ligne
            last_row = df.iloc[-1]
            
            # Vérifier la présence de valeurs manquantes
            if last_row.isnull().any():
                logger.warning(f"Valeurs manquantes dans les features pour le stock (dernière ligne)")
                return None
            
            # Sélectionner les mêmes features que pendant l'entraînement
            feature_cols = [
                'SMA_20', 'SMA_50', 'SMA_200',
                'Volume', 'Volume_SMA',
                'RSI', 'MACD', 'Volatility', 'Momentum'
            ]
            
            available_cols = [col for col in feature_cols if col in last_row.index]
            
            if available_cols:
                return last_row[available_cols].values.astype(float)
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Erreur préparation features _prepare_stock_features: {e}")
            return None
    
    def save_results(self, 
                    recommended_stocks: List[Dict[str, Any]],
                    output_file: str = 'screened_stocks.csv'):
        """Sauvegarde les résultats du screening"""
        try:
            if not recommended_stocks:
                logger.warning("Aucun stock à sauvegarder")
                return
            
            df = pd.DataFrame(recommended_stocks)
            
            # Ajouter des colonnes supplémentaires
            df['screening_date'] = datetime.now().strftime('%Y-%m-%d')
            df['model_accuracy'] = self.accuracy
            
            # Réorganiser les colonnes
            cols = ['symbol', 'current_price', 'buy_probability', 'signal_strength',
                   'rsi', 'price_vs_sma50', 'volume', 'screening_date', 'model_accuracy']
            df = df[[c for c in cols if c in df.columns]]
            
            # Sauvegarder
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Résultats sauvegardés: {output_file} ({len(df)} stocks)")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde résultats: {e}")
            return None
    
    def create_summary_report(self) -> Dict[str, Any]:
        """Crée un rapport de summary du screening"""
        report = {
            'screening_date': datetime.now().isoformat(),
            'model_accuracy': self.accuracy,
            'total_screened_stocks': len(self.screened_stocks),
            'stocks_by_signal': {},
            'average_probability': 0.0,
            'top_stocks': []
        }
        
        if self.screened_stocks:
            # Regrouper par force de signal
            for stock in self.screened_stocks:
                signal = stock['signal_strength']
                report['stocks_by_signal'][signal] = report['stocks_by_signal'].get(signal, 0) + 1
            
            # Moyenne des probabilités
            report['average_probability'] = np.mean([s['buy_probability'] for s in self.screened_stocks])
            
            # Top 5 stocks
            report['top_stocks'] = self.screened_stocks[:5]
        
        return report


def screen_sp500(lookback_years: int = 3,
                min_probability: float = 0.65,
                max_stocks: int = 15,
                model_type: str = 'random_forest') -> Dict[str, Any]:
    """
    Fonction principale pour screen les stocks du S&P 500
    
    Returns:
        Dict avec les résultats du screening
    """
    logger.info("🔍 Démarrage du screening S&P 500")
    logger.info(f"Paramètres: lookback={lookback_years} ans, min_prob={min_probability}")
    
    try:
        # Initialiser le screener
        screener = StockScreener(lookback_years=lookback_years)
        
        # Récupérer les symboles
        symbols = screener.get_sp500_symbols()
        
        # Récupérer les données (limité à 150 symboles pour la performance)
        data = screener.fetch_stock_data(symbols[:150])
        
        if not data:
            raise ValueError("Aucune donnée récupérée")
        
        # Calculer les features
        features, labels = screener.calculate_features(data)
        
        # Entraîner le modèle
        metrics = screener.train_model(model_type=model_type)
        
        # Screen les stocks
        recommended_stocks = screener.screen_stocks(
            data=data,
            min_probability=min_probability,
            max_stocks=max_stocks
        )
        
        # Sauvegarder les résultats
        output_file = f"screened_stocks_{datetime.now().strftime('%Y%m%d')}.csv"
        results_df = screener.save_results(recommended_stocks, output_file)
        
        # Créer le rapport
        report = screener.create_summary_report()
        report['metrics'] = metrics
        
        logger.info(f"✅ Screening terminé: {len(recommended_stocks)} stocks recommandés")
        
        return {
            'success': True,
            'report': report,
            'stocks': recommended_stocks,
            'metrics': metrics,
            'output_file': output_file
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur screening S&P 500: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'stocks': []
        }


def get_sp500_symbols() -> List[str]:
    """Fonction utilitaire pour récupérer les symboles du S&P 500"""
    return StockScreener.get_sp500_symbols()


def main():
    """Fonction principale pour exécuter le screening en standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Screening des actions du S&P 500")
    parser.add_argument("--lookback", type=int, default=3, 
                       help="Années de données historiques (défaut: 3)")
    parser.add_argument("--min-prob", type=float, default=0.65,
                       help="Probabilité minimale pour recommandation (défaut: 0.65)")
    parser.add_argument("--max-stocks", type=int, default=15,
                       help="Nombre maximum de stocks à recommander (défaut: 15)")
    parser.add_argument("--model", choices=['random_forest', 'gradient_boosting'],
                       default='random_forest', help="Type de modèle (défaut: random_forest)")
    
    args = parser.parse_args()
    
    logger.info("🚀 SCREENING S&P 500 - Analyse d'actions avec IA")
    logger.info("=" * 60)
    
    results = screen_sp500(
        lookback_years=args.lookback,
        min_probability=args.min_prob,
        max_stocks=args.max_stocks,
        model_type=args.model
    )
    
    if results['success']:
        report = results['report']
        stocks = results['stocks']
        
        logger.info("\n📊 RÉSULTATS DU SCREENING:")
        logger.info(f"   📅 Date: {report['screening_date'].split('T')[0]}")
        logger.info(f"   🎯 Précision du modèle: {results['metrics']['accuracy']:.3f}")
        logger.info(f"   📈 Stocks recommandés: {report['total_screened_stocks']}")
        
        if stocks:
            logger.info("\n🏆 TOP 5 ACTIONS RECOMMANDÉES:")
            for i, stock in enumerate(stocks[:5], 1):
                logger.info(f"   {i}. {stock['symbol']}:")
                logger.info(f"      Prix: ${stock['current_price']:.2f}")
                logger.info(f"      Probabilité achat: {stock['buy_probability']:.1%}")
                logger.info(f"      Signal: {stock['signal_strength']}")
                if stock['rsi']:
                    logger.info(f"      RSI: {stock['rsi']:.1f}")
                logger.info("")
        
        logger.info(f"\n💾 Résultats sauvegardés: {results['output_file']}")
        logger.info("💡 Conseil: Analysez ces actions avec le module principal pour plus de détails")
    else:
        logger.error(f"❌ Erreur: {results['error']}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    # Initialize logging for CLI execution
    try:
        from trading_algo.logging_config import init_logging
        init_logging(level=os.getenv("LOG_LEVEL", None), logfile=os.getenv("LOG_FILE", None))
    except Exception:
        # fallback for environments where package import may not work
        import logging as _logging
        _logging.basicConfig(level=_logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()