"""
Module avancé pour les prédictions et trading d'actions
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import pickle
import numpy as np
import pandas as pd
import gc
import tensorflow as tf
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import des modules internes
from trading_algo.data.data_extraction import StockDataExtractor, get_stock_overview, MacroDataExtractor
from trading_algo.visualization.dashboard import TradingDashboard
from trading_algo.models.find_best_model import ImprovedLSTMPredictorMultiOutput, ModelFinder
from trading_algo.models.stockmodeltrain import StockModelTrain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictor(StockModelTrain):
    """
    Classe avancée pour les prédictions d'actions avec IA
    Hérite de StockModelTrain et ajoute des fonctionnalités avancées
    """
    
    def __init__(self, symbol: str, period: str = "1y"):
        # Initialiser la classe parent
        super().__init__(symbol, period)
        
        # Ajouter des attributs spécifiques à StockPredictor
        self.prediction_history = []
        self.ensemble_models = []
        self.confidence_level = 0.95
        self.risk_tolerance = 0.1  # 10% de tolérance au risque
        
        logger.info(f"StockPredictor avancé initialisé pour {symbol}")
    
    def predict_future(
        self,
        days_ahead: int = 30,
        confidence_level: float = 0.95,
        scenario_analysis: bool = True,
        include_sentiment: bool = True,
        ensemble_method: str = "weighted"
    ) -> Dict[str, Any]:
        """
        Prédit les prix futurs et génère des signaux de trading.
        """
        logger.info(f"🔮 Début de la prédiction pour {days_ahead} jours à venir")
        
        try:
            # Vérifications préliminaires
            if days_ahead <= 0:
                raise ValueError("days_ahead doit être > 0")
            
            if confidence_level <= 0 or confidence_level >= 1:
                raise ValueError("confidence_level doit être entre 0 et 1")
            
            if self.model is None:
                raise ValueError("Aucun modèle entraîné disponible")
            
            if self.data is None or len(self.data) < 100:
                raise ValueError("Données historiques insuffisantes")
            
            # Préparer les données pour la prédiction
            logger.info("Préparation des données pour la prédiction...")
            
            # Utiliser les dernières données disponibles
            recent_data = self.data.tail(min(252, len(self.data)))
            
            # Préparer les features
            features_df = self._prepare_prediction_features(
                recent_data, 
                include_sentiment=include_sentiment
            )
            
            if features_df.empty or len(features_df) < 50:
                logger.warning("Features insuffisantes pour la prédiction")
                return self._get_empty_prediction_result()
            
            # Générer les prédictions
            logger.info(f"Génération des prédictions avec méthode {ensemble_method}...")
            
            predictions = self._generate_predictions(
                features_df, 
                days_ahead, 
                method=ensemble_method
            )
            
            # Calculer les intervalles de confiance
            confidence_intervals = self._calculate_confidence_intervals(
                predictions, 
                confidence_level
            )
            
            # Générer les signaux de trading
            logger.info("Génération des signaux de trading...")
            
            trading_signals = self._generate_trading_signals(
                predictions,
                confidence_intervals,
                recent_data
            )
            
            # Analyse de scénarios (optionnel)
            scenarios = None
            if scenario_analysis:
                logger.info("Analyse de scénarios...")
                scenarios = self._analyze_scenarios(
                    features_df,
                    days_ahead,
                    predictions
                )
            
            # Calculer les métriques
            logger.info("Calcul des métriques de performance...")
            
            metrics = self._calculate_prediction_metrics(
                predictions,
                trading_signals,
                recent_data
            )
            
            # Formater les résultats
            result_df = self._format_predictions_to_dataframe(
                predictions,
                confidence_intervals,
                recent_data,
                days_ahead
            )
            
            # Enregistrer l'historique
            self._save_prediction_history(result_df, metrics)
            
            logger.info(f"✅ Prédiction terminée: {len(trading_signals['signals'])} signaux générés")
            
            return {
                'predictions': result_df,
                'signals': trading_signals,
                'metrics': metrics,
                'scenarios': scenarios,
                'ensemble_method': ensemble_method,
                'confidence_level': confidence_level,
                'prediction_date': pd.Timestamp.now(),
                'horizon_days': days_ahead
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la prédiction: {str(e)}", exc_info=True)
            return self._handle_prediction_error(e)
    
    def _prepare_prediction_features(self, data: pd.DataFrame, include_sentiment: bool = True) -> pd.DataFrame:
        """Prépare les features pour la prédiction."""
        try:
            # Utiliser la méthode de la classe parent pour les indicateurs techniques
            # features_df = super().calculate_technical_indicators(data.copy())
            features_df = StockDataExtractor.calculate_technical_indicators(self, data=data)
            if features_df.empty:
                return pd.DataFrame()
            
            # Ajouter des features supplémentaires spécifiques à StockPredictor
            features_df = self._add_advanced_features(features_df)
            
            # Ajouter les données de sentiment si demandé
            if include_sentiment and hasattr(self, 'sentiment_data'):
                features_df = self._merge_sentiment_features(features_df)
            
            # Normaliser
            if hasattr(self, 'feature_scaler') and self.feature_scaler is not None:
                feature_cols = [col for col in features_df.columns if col in self.feature_scaler.feature_names_in_]
                if feature_cols:
                    features_df[feature_cols] = self.feature_scaler.transform(features_df[feature_cols])
            
            return features_df
            
        except Exception as e:
            logger.error(f"Erreur préparation features: {e}")
            return pd.DataFrame()
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des features avancées pour la prédiction."""
        try:
            # Features de momentum avancées
            df['Momentum_Change'] = df['Close'].pct_change(periods=5)
            df['Volatility_Ratio'] = df['Close'].rolling(10).std() / df['Close'].rolling(30).std()
            
            # Features de volume
            df['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # Features de tendance
            df['Trend_Strength'] = abs(df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
            
            # Remplir les NaN
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur ajout features avancées: {e}")
            return df
    
    def _generate_predictions(self, features_df: pd.DataFrame, days_ahead: int, method: str = "weighted") -> Dict:
        """Génère des prédictions avec différentes méthodes."""
        predictions = {}
        
        # Prédiction avec le modèle principal
        if self.model is not None:
            try:
                # Préparer l'input pour le modèle
                X_input = self._prepare_model_input(features_df)
                
                # Faire la prédiction
                if X_input is not None:
                    y_pred = self.model.predict(X_input, verbose=0)
                    predictions['main_model'] = y_pred[0]
            except Exception as e:
                logger.error(f"Erreur prédiction modèle principal: {e}")
        
        # Prédiction avec modèle simple (moyenne mobile)
        try:
            simple_pred = self._simple_prediction(features_df, days_ahead)
            predictions['simple_model'] = simple_pred
        except Exception as e:
            logger.error(f"Erreur prédiction simple: {e}")
        
        # Combiner les prédictions selon la méthode
        if method == "weighted" and len(predictions) > 0:
            final_pred = self._weighted_ensemble(predictions)
        elif method == "average" and len(predictions) > 0:
            final_pred = self._average_ensemble(predictions)
        else:
            # Par défaut, utiliser le modèle principal ou le simple
            if 'main_model' in predictions:
                final_pred = predictions['main_model']
            elif 'simple_model' in predictions:
                final_pred = predictions['simple_model']
            else:
                final_pred = np.zeros(days_ahead)
        
        return {
            'individual': predictions,
            'ensemble': final_pred,
            'method': method
        }
    
    def _prepare_model_input(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prépare l'input pour le modèle LSTM."""
        try:
            # Prendre les dernières séquences
            sequence_length = 60  # Même que lors de l'entraînement
            if len(features_df) < sequence_length:
                sequence_length = len(features_df)
            
            recent_features = features_df.iloc[-sequence_length:].values
            
            # Normaliser si scaler disponible
            if hasattr(self, 'feature_scaler') and self.feature_scaler is not None:
                recent_features = self.feature_scaler.transform(recent_features)
            
            # Reshape pour LSTM: (1, sequence_length, n_features)
            return recent_features.reshape(1, sequence_length, -1)
            
        except Exception as e:
            logger.error(f"Erreur préparation input modèle: {e}")
            return None
    
    def _simple_prediction(self, features_df: pd.DataFrame, days_ahead: int) -> np.ndarray:
        """Prédiction simple basée sur la moyenne mobile."""
        try:
            recent_prices = features_df['Close'].values if 'Close' in features_df.columns else features_df.iloc[:, 0].values
            
            # Calculer la tendance
            if len(recent_prices) > 10:
                trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            else:
                trend = 0
            
            # Générer des prédictions basées sur la tendance
            last_price = recent_prices[-1]
            predictions = []
            
            for i in range(days_ahead):
                next_price = last_price + trend * (i + 1)
                # Ajouter un peu de bruit
                noise = np.random.normal(0, abs(trend) * 0.5)
                predictions.append(next_price + noise)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Erreur prédiction simple: {e}")
            return np.zeros(days_ahead)
    
    def _weighted_ensemble(self, predictions: Dict) -> np.ndarray:
        """Combine les prédictions avec des poids."""
        weights = {
            'main_model': 0.7,
            'simple_model': 0.3
        }
        
        weighted_sum = None
        total_weight = 0
        
        for model_name, pred in predictions.items():
            if model_name in weights:
                weight = weights[model_name]
                if weighted_sum is None:
                    weighted_sum = pred * weight
                else:
                    # Assurer la même longueur
                    min_len = min(len(weighted_sum), len(pred))
                    weighted_sum[:min_len] += pred[:min_len] * weight
                
                total_weight += weight
        
        if weighted_sum is not None and total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Retourner la première prédiction disponible
            return list(predictions.values())[0] if predictions else np.array([])
    
    def _average_ensemble(self, predictions: Dict) -> np.ndarray:
        """Moyenne des prédictions."""
        all_preds = list(predictions.values())
        
        if not all_preds:
            return np.array([])
        
        # Trouver la longueur minimale
        min_len = min(len(p) for p in all_preds)
        
        # Moyenne des prédictions
        avg_pred = np.zeros(min_len)
        for pred in all_preds:
            avg_pred += pred[:min_len]
        
        return avg_pred / len(all_preds)
    
    def _calculate_confidence_intervals(self, predictions: Dict, confidence_level: float = 0.95) -> Dict:
        """Calcule les intervalles de confiance."""
        try:
            individual_preds = predictions.get('individual', {})
            
            if not individual_preds:
                return {}
            
            # Collecter toutes les prédictions
            all_preds = list(individual_preds.values())
            
            if len(all_preds) < 2:
                return {}
            
            # Calculer la moyenne et l'écart-type
            min_len = min(len(p) for p in all_preds)
            stacked = np.array([p[:min_len] for p in all_preds])
            
            mean_pred = np.mean(stacked, axis=0)
            std_pred = np.std(stacked, axis=0)
            
            # Z-score pour l'intervalle de confiance
            if confidence_level == 0.95:
                z_score = 1.96
            elif confidence_level == 0.99:
                z_score = 2.58
            else:
                z_score = 1.96  # Par défaut
            
            lower_bound = mean_pred - z_score * std_pred
            upper_bound = mean_pred + z_score * std_pred
            
            return {
                'mean': mean_pred,
                'std': std_pred,
                'lower': lower_bound,
                'upper': upper_bound,
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul intervalles confiance: {e}")
            return {}
    
    def _generate_trading_signals(self, predictions: Dict, confidence_intervals: Dict, recent_data: pd.DataFrame) -> Dict:
        """Génère des signaux de trading."""
        try:
            if 'Close' not in recent_data.columns:
                return {'error': 'Données de prix manquantes'}
            
            current_price = recent_data['Close'].iloc[-1]
            ensemble_pred = predictions.get('ensemble')
            
            if ensemble_pred is None or len(ensemble_pred) == 0:
                return {'error': 'Prédictions manquantes'}
            
            signals = []
            
            # Analyser différents horizons
            for horizon in [1, 5, 10, 20]:
                if horizon <= len(ensemble_pred):
                    pred_price = ensemble_pred[horizon - 1]
                    expected_return = (pred_price - current_price) / current_price
                    
                    # Déterminer le signal
                    signal = self._determine_signal(expected_return, horizon)
                    
                    # Calculer les niveaux de risque
                    stop_loss, take_profit = self._calculate_risk_levels(
                        current_price, 
                        signal, 
                        horizon
                    )
                    
                    signals.append({
                        'horizon': horizon,
                        'predicted_price': pred_price,
                        'expected_return': expected_return,
                        'signal': signal,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': confidence_intervals.get('confidence_level', 0.95)
                    })
            
            # Recommandation globale
            overall_signal = self._calculate_overall_signal(signals)
            
            return {
                'signals': signals,
                'overall_signal': overall_signal,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur génération signaux: {e}")
            return {'error': str(e)}
    
    def _determine_signal(self, expected_return: float, horizon: int) -> str:
        """Détermine le signal de trading."""
        # Seuils adaptés à l'horizon
        if horizon <= 5:
            threshold = 0.02  # 2% pour court terme
        elif horizon <= 20:
            threshold = 0.05  # 5% pour moyen terme
        else:
            threshold = 0.08  # 8% pour long terme
        
        if expected_return > threshold:
            return "BUY"
        elif expected_return < -threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_risk_levels(self, current_price: float, signal: str, horizon: int) -> Tuple[float, float]:
        """Calcule les niveaux de stop loss et take profit."""
        # Multiplicateurs basés sur l'horizon et le signal
        if horizon <= 5:
            risk_multiplier = 1.0
        elif horizon <= 20:
            risk_multiplier = 1.5
        else:
            risk_multiplier = 2.0
        
        if signal == "BUY":
            stop_loss = current_price * (1 - 0.02 * risk_multiplier)
            take_profit = current_price * (1 + 0.04 * risk_multiplier)
        elif signal == "SELL":
            stop_loss = current_price * (1 + 0.02 * risk_multiplier)
            take_profit = current_price * (1 - 0.04 * risk_multiplier)
        else:  # HOLD
            stop_loss = current_price * (1 - 0.01 * risk_multiplier)
            take_profit = current_price * (1 + 0.02 * risk_multiplier)
        
        return stop_loss, take_profit
    
    def _calculate_overall_signal(self, signals: List[Dict]) -> Dict:
        """Calcule le signal global."""
        if not signals:
            return {'signal': 'NEUTRAL', 'confidence': 0.0}
        
        buy_count = sum(1 for s in signals if s['signal'] == 'BUY')
        sell_count = sum(1 for s in signals if s['signal'] == 'SELL')
        hold_count = sum(1 for s in signals if s['signal'] == 'HOLD')
        
        total = len(signals)
        
        if buy_count > sell_count and buy_count > hold_count:
            signal = 'BUY'
            confidence = buy_count / total
        elif sell_count > buy_count and sell_count > hold_count:
            signal = 'SELL'
            confidence = sell_count / total
        else:
            signal = 'HOLD'
            confidence = hold_count / total
        
        return {
            'signal': signal,
            'confidence': confidence,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'total_signals': total
        }
    
    def _analyze_scenarios(self, features_df: pd.DataFrame, days_ahead: int, predictions: Dict) -> Dict:
        """Analyse de scénarios."""
        scenarios = {
            'optimistic': {'multiplier': 1.2, 'probability': 0.25},
            'neutral': {'multiplier': 1.0, 'probability': 0.50},
            'pessimistic': {'multiplier': 0.8, 'probability': 0.25}
        }
        
        scenario_results = {}
        
        for name, params in scenarios.items():
            adjusted_pred = predictions.get('ensemble', np.zeros(days_ahead)) * params['multiplier']
            
            scenario_results[name] = {
                'predictions': adjusted_pred,
                'multiplier': params['multiplier'],
                'probability': params['probability'],
                'expected_return': np.mean(adjusted_pred) / self.current_price - 1 if self.current_price > 0 else 0
            }
        
        return {
            'scenarios': scenario_results,
            'recommended_scenario': max(
                scenario_results.items(),
                key=lambda x: x[1]['expected_return'] * x[1]['probability']
            )[0]
        }
    
    def _calculate_prediction_metrics(self, predictions: Dict, signals: Dict, historical_data: pd.DataFrame) -> Dict:
        """Calcule les métriques de performance."""
        metrics = {}
        
        try:
            # Précision des prédictions (si nous avons des données historiques suffisantes)
            if len(historical_data) > 30:
                # Backtesting simple
                actual_returns = historical_data['Close'].pct_change().dropna()
                
                # Métriques de base
                metrics['historical_volatility'] = actual_returns.std() * np.sqrt(252)
                metrics['sharpe_ratio'] = actual_returns.mean() / actual_returns.std() * np.sqrt(252) if actual_returns.std() > 0 else 0
            
            # Métriques des signaux
            if 'signals' in signals and signals['signals']:
                signal_metrics = {
                    'total_signals': len(signals['signals']),
                    'buy_signals': sum(1 for s in signals['signals'] if s.get('signal') == 'BUY'),
                    'sell_signals': sum(1 for s in signals['signals'] if s.get('signal') == 'SELL'),
                    'hold_signals': sum(1 for s in signals['signals'] if s.get('signal') == 'HOLD'),
                    'avg_expected_return': np.mean([s.get('expected_return', 0) for s in signals['signals']])
                }
                metrics.update(signal_metrics)
            
            # Score de confiance
            metrics['confidence_score'] = self._calculate_confidence_score(predictions)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques: {e}")
            return {'error': str(e)}
    
    def _calculate_confidence_score(self, predictions: Dict) -> float:
        """Calcule un score de confiance."""
        try:
            individual_preds = predictions.get('individual', {})
            
            if not individual_preds:
                return 0.5
            
            # Variance entre les modèles
            all_preds = list(individual_preds.values())
            if len(all_preds) < 2:
                return 0.6
            
            # Calculer la cohérence
            min_len = min(len(p) for p in all_preds)
            first_pred = all_preds[0][:min_len]
            
            correlations = []
            for pred in all_preds[1:]:
                if len(pred) >= min_len:
                    corr = np.corrcoef(first_pred, pred[:min_len])[0, 1]
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)
            
            avg_correlation = np.mean(correlations) if correlations else 0
            
            # Score basé sur la corrélation
            return min(1.0, max(0.0, avg_correlation))
            
        except:
            return 0.5
    
    def _format_predictions_to_dataframe(self, predictions: Dict, confidence_intervals: Dict, 
                                         historical_data: pd.DataFrame, days_ahead: int) -> pd.DataFrame:
        """Formate les prédictions en DataFrame."""
        try:
            # Créer les dates futures
            last_date = historical_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=days_ahead,
                freq='B'
            )
            
            # Créer le DataFrame
            result_df = pd.DataFrame(index=future_dates)
            
            # Ajouter les prédictions
            ensemble_pred = predictions.get('ensemble')
            if ensemble_pred is not None and len(ensemble_pred) == days_ahead:
                result_df['Predicted_Close'] = ensemble_pred
            
            # Ajouter les intervalles de confiance
            if confidence_intervals:
                if 'lower' in confidence_intervals and 'upper' in confidence_intervals:
                    lower = confidence_intervals['lower']
                    upper = confidence_intervals['upper']
                    
                    if len(lower) == days_ahead and len(upper) == days_ahead:
                        result_df['CI_Lower'] = lower
                        result_df['CI_Upper'] = upper
            
            # Ajouter les rendements attendus
            if 'Close' in historical_data.columns and 'Predicted_Close' in result_df.columns:
                current_price = historical_data['Close'].iloc[-1]
                result_df['Expected_Return'] = (result_df['Predicted_Close'] - current_price) / current_price
            
            return result_df
            
        except Exception as e:
            logger.error(f"Erreur formatage prédictions: {e}")
            return pd.DataFrame()
    
    def _get_empty_prediction_result(self) -> Dict:
        """Retourne un résultat vide."""
        return {
            'predictions': pd.DataFrame(),
            'signals': {'overall_signal': 'NEUTRAL', 'signals': []},
            'metrics': {},
            'scenarios': None,
            'error': 'Prédiction impossible'
        }
    
    def _handle_prediction_error(self, error: Exception) -> Dict:
        """Gère les erreurs de prédiction."""
        return {
            'predictions': pd.DataFrame(),
            'signals': {'overall_signal': 'ERROR', 'signals': []},
            'metrics': {'error': str(error)},
            'scenarios': None,
            'error': str(error)
        }
    
    def _save_prediction_history(self, predictions_df: pd.DataFrame, metrics: Dict):
        """Sauvegarde l'historique des prédictions."""
        try:
            if predictions_df.empty:
                return
            
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': self.symbol,
                'predictions': predictions_df.to_dict('records'),
                'metrics': metrics
            }
            
            self.prediction_history.append(history_entry)
            
            # Limiter l'historique à 100 entrées
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde historique: {e}")
    
    def get_prediction_history(self) -> List[Dict]:
        """Retourne l'historique des prédictions."""
        return self.prediction_history
    
    def analyze_stock_advanced(self) -> Dict[str, Any]:
        """
        Analyse avancée d'une action avec prédictions détaillées
        """
        logger.info(f"🔍 Analyse avancée de {self.symbol}")
        
        try:
            # Récupérer les données
            if not self.fetch_data():
                return {"error": "Échec de la récupération des données"}
            
            # Entraîner le modèle si nécessaire
            if self.model is None:
                logger.info("Entraînement du modèle...")
                if not self.train(epochs=30, lookback_days=60):
                    return {"error": "Échec de l'entraînement"}
            
            # Générer les prédictions
            logger.info("Génération des prédictions avancées...")
            
            # Prédictions à différents horizons
            predictions_1d = self.predict_future(days_ahead=1)
            predictions_5d = self.predict_future(days_ahead=5)
            predictions_20d = self.predict_future(days_ahead=20)
            predictions_90d = self.predict_future(days_ahead=90)
            
            # Calculer le score
            self.trading_score = StockModelTrain.calculate_trading_score(predictions_90d)
            self.recommendation = StockModelTrain.generate_recommendation(self.trading_score)
            
            # Compiler les résultats
            results = {
                "symbol": self.symbol,
                "current_price": self.current_price,
                "trading_score": self.trading_score,
                "recommendation": self.recommendation,
                "predictions": {
                    "1d": predictions_1d,
                    "5d": predictions_5d,
                    "20d": predictions_20d,
                    "90d": predictions_90d
                },
                "technical_indicators": self._get_technical_summary(),
                "market_context": self._get_market_context(),
                "analysis_date": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Analyse avancée terminée pour {self.symbol}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse avancée: {e}", exc_info=True)
            return {"error": str(e), "symbol": self.symbol}
    
    def _get_technical_summary(self) -> Dict:
        """Récupère un résumé des indicateurs techniques."""
        try:
            if self.features is None or self.features.empty:
                return {}
            
            last_row = self.features.iloc[-1]
            
            summary = {}
            
            # Indicateurs clés
            key_indicators = ['RSI', 'MACD', 'SMA_20', 'SMA_50', 'SMA_200', 'ATR', 'BB_Width']
            
            for indicator in key_indicators:
                if indicator in last_row:
                    summary[indicator] = float(last_row[indicator])
            
            return summary
            
        except Exception as e:
            logger.error(f"Erreur résumé technique: {e}")
            return {}
    
    def _get_market_context(self) -> Dict:
        """Récupère le contexte de marché."""
        try:
            context = {
                "market_indicators": self.macro_extractor.get_market_indicators(),
                "economic_indicators": self.macro_extractor.get_economic_indicators("US"),
                "commodities": self.macro_extractor.get_commodity_prices()
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Erreur contexte marché: {e}")
            return {}


def main():
    """Fonction principale pour tester StockPredictor."""
    print("🚀 STOCK PREDICTOR - Prédictions Avancées")
    print("=" * 60)
    
    # Test avec une action
    symbol = "AAPL"
    
    try:
        print(f"\n📈 Analyse avancée de {symbol}")
        print("-" * 40)
        
        predictor = StockPredictor(symbol, period="3y")
        results = predictor.analyze_stock_advanced()
        
        if 'error' in results:
            print(f"❌ Erreur: {results['error']}")
            return
        
        print(f"✅ Analyse terminée!")
        print(f"   Prix actuel: ${results['current_price']:.2f}")
        print(f"   Score de trading: {results['trading_score']:.1f}/10")
        print(f"   Recommandation: {results['recommendation']}")
        
        # Afficher quelques prédictions
        predictions_1d = results['predictions']['1d']
        if 'predictions' in predictions_1d and not predictions_1d['predictions'].empty:
            print(f"\n📊 Prédiction 1 jour:")
            pred_df = predictions_1d['predictions']
            if not pred_df.empty:
                print(f"   Prix prévu: ${pred_df.iloc[0]['Predicted_Close']:.2f}")
        
        # Signaux de trading
        if 'signals' in predictions_1d:
            signals = predictions_1d['signals']
            if 'overall_signal' in signals:
                print(f"   Signal: {signals['overall_signal']}")
        
        print("\n" + "=" * 60)
        print("✅ Test de StockPredictor terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()