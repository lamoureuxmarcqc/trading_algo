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
from trading_algo.visualization.symbol_dashboard import AdvancedTradingDashboard
from trading_algo.models.base_model import ImprovedLSTMPredictorMultiOutput
from trading_algo.models.stockmodeltrain import StockModelTrain
from trading_algo.risk.risk_manager import RiskManager
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor(StockModelTrain):
    def __init__(self, symbol: str, period: str = "1y"):
        super().__init__(symbol, period)
        self.prediction_history = []
        self.ensemble_models = []
        self.confidence_level = 0.95
        self.risk_tolerance = 0.1
        self.risk_manager = RiskManager()
        self.risk_metrics = None
        self.market_data_extractor = StockDataExtractor()  # Pour récupérer les données de marché
        logger.info(f"StockPredictor avancé initialisé pour {symbol}")

    def analyze_stock_advanced(self) -> Dict[str, Any]:
        logger.info(f"🔍 Analyse avancée de {self.symbol}")
        try:
            if not self.fetch_data():
                return {"error": "Échec de la récupération des données"}

            if self.model is None:
                logger.info("Entraînement du modèle...")
                if not self.train(epochs=30, lookback_days=60):
                    return {"error": "Échec de l'entraînement"}

            logger.info("Génération des prédictions avancées...")
            predictions_dict = self.generate_predictions()

            # Initialisation des métriques de risque
            risk_metrics = {}

            if self.data is not None and not self.data.empty:
                returns = self.data['Close'].pct_change().dropna()

                # Récupération des données de marché pour le bêta (S&P 500)
                market_data = self.market_data_extractor.get_historical_data('^GSPC', self.period)
                if market_data is not None and not market_data.empty:
                    market_returns = market_data['Close'].pct_change().dropna()
                    # Alignement des séries
                    common_index = returns.index.intersection(market_returns.index)
                    stock_returns_aligned = returns.loc[common_index]
                    market_returns_aligned = market_returns.loc[common_index]
                    risk_metrics['beta'] = self.risk_manager.calculate_beta(
                        stock_returns_aligned, market_returns_aligned
                    )

                # Métriques de risque standards
                risk_metrics['sharpe_ratio'] = self.risk_manager.calculate_sharpe_ratio(returns)
                risk_metrics['max_drawdown'] = self.risk_manager.calculate_max_drawdown(self.data['Close'])
                risk_metrics['value_at_risk'] = self.risk_manager.calculate_value_at_risk(
                    returns, self.confidence_level
                )

                # Calcul unique des niveaux ATR (basé sur les données actuelles)
                atr_levels = self.risk_manager.calculate_atr_levels(self.data)
                # Support dict (new) or tuple/list (legacy)
                if isinstance(atr_levels, dict):
                    stop_loss_base = atr_levels.get('stop_loss')
                    take_profit_base = atr_levels.get('take_profit')
                else:
                    try:
                        stop_loss_base, take_profit_base = atr_levels
                    except Exception:
                        stop_loss_base = None
                        take_profit_base = None

                # Préparation des dictionnaires pour chaque horizon de prédiction
                stop_loss_levels = {}
                take_profit_levels = {}
                risk_reward_ratios = {}
                position_sizes = {}

                # Paramètres pour le sizing
                account_balance = 100000.0  # À rendre paramétrable si besoin
                risk_tolerance = getattr(self, 'risk_tolerance', 0.02)  # 2% par défaut

                # Si generate_predictions renvoie la structure classique { '1d': price, ... }
                # itérer sur items() fonctionne. If generate_predictions returns other structure,
                # use the 'predictions' entry if available.
                preds_iter = predictions_dict
                if isinstance(predictions_dict, dict) and 'ensemble' in predictions_dict and 'individual' in predictions_dict:
                    # prefer the simple mapping produced by legacy generate_predictions if present
                    # fallback to individual models otherwise
                    if isinstance(predictions_dict.get('individual'), dict) and predictions_dict.get('individual'):
                        preds_iter = {k: None for k in predictions_dict['individual'].keys()}  # keep keys only
                    else:
                        preds_iter = predictions_dict.get('ensemble', {})

                for horizon, pred_price in (preds_iter.items() if isinstance(preds_iter, dict) else []):
                    if pred_price is not None and self.current_price > 0:
                        if stop_loss_base is not None and take_profit_base is not None:
                            entry = self.current_price
                            stop_loss_levels[horizon] = round(stop_loss_base, 2)
                            take_profit_levels[horizon] = round(take_profit_base, 2)
                            risk_reward_ratios[horizon] = self.risk_manager.risk_reward_ratio(
                                entry, stop_loss_base, pred_price
                            )
                            position_sizes[horizon] = self.risk_manager.suggest_position_size(
                                account_balance, entry, stop_loss_base, risk_tolerance
                            )

                # Ajout des dictionnaires dans risk_metrics
                risk_metrics['stop_loss_levels'] = stop_loss_levels
                risk_metrics['take_profit_levels'] = take_profit_levels
                risk_metrics['risk_reward_ratios'] = risk_reward_ratios
                risk_metrics['suggested_position_sizes'] = position_sizes

            # Sauvegarde pour usage ultérieur
            self.risk_metrics = risk_metrics

            # Calcul du score et de la recommandation
            self.trading_score = self.calculate_trading_score(predictions_dict)
            self.recommendation = self.generate_recommendation(self.trading_score)

            # Construction du résultat final
            results = {
                "symbol": self.symbol,
                "current_price": self.current_price,
                "trading_score": self.trading_score,
                "recommendation": self.recommendation,
                "predictions": predictions_dict,
                "technical_indicators": self._get_technical_summary(),
                "risk_metrics": risk_metrics,
                "market_context": self._get_market_context(),
                "analysis_date": datetime.now().isoformat()
            }

            logger.info(f"✅ Analyse avancée terminée pour {self.symbol}")
            return results

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse avancée avancée de {self.symbol}: {e}")
            return {"error": str(e)}
       
    def _log_technical_indicators(self):
        """Affiche les dernières valeurs des indicateurs techniques clés"""
        if self.features is None or self.features.empty:
            logger.warning("Aucune donnée technique disponible")
            return

        last_row = self.features.iloc[-1]
        indicators = {
            'RSI (14)': last_row.get('RSI', 'N/A'),
            'MACD': last_row.get('MACD', 'N/A'),
            'Signal': last_row.get('MACD_Signal', 'N/A'),
            'SMA 20': last_row.get('SMA_20', 'N/A'),
            'SMA 50': last_row.get('SMA_50', 'N/A'),
            'SMA 200': last_row.get('SMA_200', 'N/A'),
            'ATR': last_row.get('ATR', 'N/A'),
            'Volume': last_row.get('Volume', 'N/A'),
            'Momentum 10j': last_row.get('Momentum_10', 'N/A')
        }

        logger.info("📊 Indicateurs techniques récents :")
        for name, value in indicators.items():
            if value != 'N/A' and not pd.isna(value):
                if 'SMA' in name or 'ATR' in name:
                    logger.info(f" {name}: {value:.2f}")
                elif 'RSI' in name:
                    status = "Surachat ⚠️" if value > 70 else "Survente ✅" if value < 30 else "Neutre"
                    logger.info(f" {name}: {value:.2f} ({status})")
                elif 'MACD' in name and value != 'N/A':
                    logger.info(f" {name}: {value:.4f}")
                else:
                    logger.info(f" {name}: {value}")

    def predict_future(
        self,
        days_ahead: int = 30,
        confidence_level: float = 0.95,
        scenario_analysis: bool = True,
        include_sentiment: bool = True,
        ensemble_method: str = "weighted",
        account_balance: float = 100000.0  # Ajout pour le sizing de position
    ) -> Dict[str, Any]:
        """
        Prédit les prix futurs et génère des signaux de trading avec gestion des risques intégrée.
        """
        logger.info(f"🔮 Début de la prédiction pour {days_ahead} jours à venir")

        try:
            if days_ahead <= 0:
                raise ValueError("days_ahead doit être > 0")
            if confidence_level <= 0 or confidence_level >= 1:
                raise ValueError("confidence_level doit être entre 0 et 1")
            if self.model is None:
                raise ValueError("Aucun modèle entraîné disponible")
            if self.data is None or len(self.data) < 100:
                raise ValueError("Données historiques insuffisantes")

            logger.info("Préparation des données pour la prédiction...")
            recent_data = self.data.tail(min(252, len(self.data)))

            features_df = self._prepare_prediction_features(
                recent_data,
                include_sentiment=include_sentiment
            )

            if features_df.empty or len(features_df) < 50:
                logger.warning("Features insuffisantes pour la prédiction")
                return self._get_empty_prediction_result()

            logger.info(f"Génération des prédictions avec méthode {ensemble_method}...")
            predictions = self._generate_predictions(
                features_df,
                days_ahead,
                method=ensemble_quality
            )

            # --- Interpolation des prédictions ponctuelles ---
            try:
                point_preds = self.generate_predictions()
                horizons = []
                prices = []
                mapping = {'1d': 1, '5d': 5, '10d': 10, '20d': 20, '30d': 30, '90d': 90}
                for key, day in mapping.items():
                    if key in point_preds and point_preds[key] is not None:
                        horizons.append(day)
                        prices.append(point_preds[key])
                if len(horizons) >= 2:
                    all_days = np.arange(1, days_ahead + 1)
                    interpolated = np.interp(all_days, horizons, prices)
                    predictions['ensemble'] = interpolated
                    logger.info(f"Prédictions interpolées à partir de {len(horizons)} points")
                else:
                    logger.warning("Pas assez de points pour l'interpolation, utilisation des prédictions brutes")
            except Exception as e:
                logger.warning(f"Erreur lors de l'interpolation: {e}, utilisation des prédictions brutes")

            confidence_intervals = self._calculate_confidence_intervals(
                predictions,
                confidence_level
            )

            logger.info("Génération des signaux de trading avec gestion des risques...")
            trading_signals = self._generate_trading_signals(
                predictions,
                confidence_intervals,
                recent_data,
                account_balance
            )

            scenarios = None
            if scenario_analysis:
                logger.info("Analyse de scénarios...")
                scenarios = self._analyze_scenarios(
                    features_df,
                    days_ahead,
                    predictions
                )

            logger.info("Calcul des métriques de performance...")
            metrics = self._calculate_prediction_metrics(
                predictions,
                trading_signals,
                recent_data
            )

            result_df = self._format_predictions_to_dataframe(
                predictions,
                confidence_intervals,
                recent_data,
                days_ahead
            )

            self._log_technical_indicators()

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
        try:
            if self.features is None or self.features.empty:
                logger.warning("Features non disponibles, calcul à partir des données...")
                features_df = self.data_extractor.calculate_technical_indicators(data.copy())
            else:
                features_df = self.features.copy()

            if include_sentiment and hasattr(self, 'sentiment_data'):
                pass

            if not hasattr(self.feature_scaler, 'feature_names_in_'):
                logger.error("Scaler non entraîné")
                return pd.DataFrame()

            expected_cols = list(self.feature_scaler.feature_names_in_)
            missing = set(expected_cols) - set(features_df.columns)
            if missing:
                logger.error(f"Colonnes manquantes : {missing}")
                return pd.DataFrame()

            features_df = features_df[expected_cols]

            scaled = self.feature_scaler.transform(features_df)
            if np.any(np.isnan(scaled)) or np.any(np.isinf(scaled)):
                logger.warning("NaN/inf dans les données normalisées, remplacement par 0")
                scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

            result_df = pd.DataFrame(
                scaled.astype(np.float32),
                index=features_df.index,
                columns=expected_cols
            )
            return result_df

        except Exception as e:
            logger.error(f"Erreur préparation features: {e}")
            return pd.DataFrame()

    def _generate_predictions(self, features_df: pd.DataFrame, days_ahead: int, method: str = "weighted") -> Dict:
        """Génère des prédictions avec différentes méthodes."""
        predictions = {}

        if self.model is not None:
            try:
                X_input = self._prepare_model_input(features_df)
                if X_input is not None:
                    y_pred = self.model.predict(X_input, verbose=0)
                    base_pred = y_pred[0, 0] if y_pred.shape[1] > 0 else 0
                    predictions['main_model'] = np.full(days_ahead, base_pred)
            except Exception as e:
                logger.error(f"Erreur prédiction modèle principal: {e}")

        try:
            simple_pred = self._simple_prediction(features_df, days_ahead)
            predictions['simple_model'] = simple_pred
        except Exception as e:
            logger.error(f"Erreur prédiction simple: {e}")

        if method == "weighted" and len(predictions) > 0:
            final_pred = self._weighted_ensemble(predictions)
        elif method == "average" and len(predictions) > 0:
            final_pred = self._average_ensemble(predictions)
        else:
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
        try:
            sequence_length = getattr(self, 'lookback_days', 60)
            if len(features_df) < sequence_length:
                logger.warning(f"Pas assez de données ({len(features_df)}), padding avec la première valeur disponible")
                pad_len = sequence_length - len(features_df)
                first_row = features_df.iloc[:1].values
                pad_values = np.repeat(first_row, pad_len, axis=0)
                features_array = np.vstack([pad_values, features_df.values])
            else:
                features_array = features_df.iloc[-sequence_length:].values

            if hasattr(self, 'feature_scaler') and self.feature_scaler is not None:
                features_array = self.feature_scaler.transform(features_array)

            return features_array.reshape(1, sequence_length, -1)

        except Exception as e:
            logger.error(f"Erreur préparation input modèle: {e}")
            return None

    def _simple_prediction(self, features_df: pd.DataFrame, days_ahead: int) -> np.ndarray:
        try:
            recent_prices = features_df['Close'].values if 'Close' in features_df.columns else features_df.iloc[:, 0].values
            if len(recent_prices) > 10:
                trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            else:
                trend = 0
            last_price = recent_prices[-1]
            predictions = []
            for i in range(days_ahead):
                next_price = last_price + trend * (i + 1)
                noise = np.random.normal(0, abs(trend) * 0.5)
                predictions.append(next_price + noise)
            return np.array(predictions)
        except Exception as e:
            logger.error(f"Erreur prédiction simple: {e}")
            return np.zeros(days_ahead)

    def _weighted_ensemble(self, predictions: Dict) -> np.ndarray:
        weights = {'main_model': 0.7, 'simple_model': 0.3}
        weighted_sum = None
        total_weight = 0
        for model_name, pred in predictions.items():
            if model_name in weights:
                weight = weights[model_name]
                if weighted_sum is None:
                    weighted_sum = pred * weight
                else:
                    min_len = min(len(weighted_sum), len(pred))
                    weighted_sum[:min_len] += pred[:min_len] * weight
                total_weight += weight
        if weighted_sum is not None and total_weight > 0:
            return weighted_sum / total_weight
        else:
            return list(predictions.values())[0] if predictions else np.array([])

    def _average_ensemble(self, predictions: Dict) -> np.ndarray:
        all_preds = list(predictions.values())
        if not all_preds:
            return np.array([])
        min_len = min(len(p) for p in all_preds)
        avg_pred = np.zeros(min_len)
        for pred in all_preds:
            avg_pred += pred[:min_len]
        return avg_pred / len(all_preds)

    def _calculate_confidence_intervals(self, predictions: Dict, confidence_level: float = 0.95) -> Dict:
        try:
            individual_preds = predictions.get('individual', {})
            if not individual_preds or len(individual_preds) < 2:
                return {}
            all_preds = list(individual_preds.values())
            min_len = min(len(p) for p in all_preds)
            stacked = np.array([p[:min_len] for p in all_preds])
            mean_pred = np.mean(stacked, axis=0)
            std_pred = np.std(stacked, axis=0)
            z_score = 1.96 if confidence_level == 0.95 else 2.58 if confidence_level == 0.99 else 1.96
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

    def _generate_trading_signals(self, predictions: Dict, confidence_intervals: Dict, recent_data: pd.DataFrame, account_balance: float) -> Dict:
        try:
            if 'Close' not in recent_data.columns:
                return {'error': 'Données de prix manquantes'}
            current_price = recent_data['Close'].iloc[-1]
            ensemble_pred = predictions.get('ensemble')
            if ensemble_pred is None or len(ensemble_pred) == 0:
                return {'error': 'Prédictions manquantes'}
            signals = []
            for horizon in [1, 5, 10, 20]:
                if horizon <= len(ensemble_pred):
                    pred_price = ensemble_pred[horizon - 1]
                    expected_return = (pred_price - current_price) / current_price
                    signal = self._determine_signal(expected_return, horizon)
                    stop_loss, take_profit = self._calculate_risk_levels(current_price, signal, horizon)
                    risk_reward = self.risk_manager.risk_reward_ratio(current_price, stop_loss, take_profit)
                    position_size = self.risk_manager.suggest_position_size(
                        account_balance, current_price, stop_loss, self.risk_tolerance
                    )
                    signals.append({
                        'horizon': horizon,
                        'predicted_price': pred_price,
                        'expected_return': expected_return,
                        'signal': signal,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_reward_ratio': risk_reward,
                        'suggested_position_size': position_size,
                        'confidence': confidence_intervals.get('confidence_level', 0.95)
                    })
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
        if horizon <= 5:
            threshold = 0.02
        elif horizon <= 20:
            threshold = 0.05
        else:
            threshold = 0.08
        if expected_return > threshold:
            return "BUY"
        elif expected_return < -threshold:
            return "SELL"
        else:
            return "HOLD"

    def _calculate_risk_levels(self, current_price: float, signal: str, horizon: int) -> Tuple[float, float]:
        # Utilisation de l'ATR pour des niveaux dynamiques
        atr_multiplier_stop = 1.5 if horizon <= 5 else 2.0 if horizon <= 20 else 2.5
        atr_multiplier_take = 2.5 if horizon <= 5 else 3.0 if horizon <= 20 else 4.0

        levels = self.risk_manager.calculate_atr_levels(self.data)
        atr = None
        if isinstance(levels, dict):
            atr = levels.get('atr')
        else:
            # legacy tuple/list support: try first element as atr
            try:
                atr = float(levels[0]) if len(levels) >= 1 else None
            except Exception:
                atr = None

        if atr is not None and atr > 0:
            stop_loss = current_price - (atr * atr_multiplier_stop)
            take_profit = current_price + (atr * atr_multiplier_take)
        else:
            # Fallback sur pourcentages fixes si ATR non disponible
            risk_multiplier = 1.0 if horizon <= 5 else 1.5 if horizon <= 20 else 2.0
            if signal == "BUY":
                stop_loss = current_price * (1 - 0.02 * risk_multiplier)
                take_profit = current_price * (1 + 0.04 * risk_multiplier)
            elif signal == "SELL":
                stop_loss = current_price * (1 + 0.02 * risk_multiplier)
                take_profit = current_price * (1 - 0.04 * risk_multiplier)
            else:
                stop_loss = current_price * (1 - 0.01 * risk_multiplier)
                take_profit = current_price * (1 + 0.02 * risk_multiplier)

        return stop_loss, take_profit

    def _calculate_overall_signal(self, signals: List[Dict]) -> Dict:
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
        metrics = {}
        try:
            if len(historical_data) > 30:
                actual_returns = historical_data['Close'].pct_change().dropna()
                metrics['historical_volatility'] = actual_returns.std() * np.sqrt(252)
                metrics['sharpe_ratio'] = actual_returns.mean() / actual_returns.std() * np.sqrt(252) if actual_returns.std() > 0 else 0
            if 'signals' in signals and signals['signals']:
                signal_metrics = {
                    'total_signals': len(signals['signals']),
                    'buy_signals': sum(1 for s in signals['signals'] if s.get('signal') == 'BUY'),
                    'sell_signals': sum(1 for s in signals['signals'] if s.get('signal') == 'SELL'),
                    'hold_signals': sum(1 for s in signals['signals'] if s.get('signal') == 'HOLD'),
                    'avg_expected_return': np.mean([s.get('expected_return', 0) for s in signals['signals']]),
                    'avg_risk_reward_ratio': np.mean([s.get('risk_reward_ratio', 0) for s in signals['signals']])
                }
                metrics.update(signal_metrics)
            metrics['confidence_score'] = self._calculate_confidence_score(predictions)
            return metrics
        except Exception as e:
            logger.error(f"Erreur calcul métriques: {e}")
            return {'error': str(e)}

    def _calculate_confidence_score(self, predictions: Dict) -> float:
        try:
            individual_preds = predictions.get('individual', {})
            if not individual_preds or len(individual_preds) < 2:
                return 0.5
            all_preds = list(individual_preds.values())
            min_len = min(len(p) for p in all_preds)
            first_pred = all_preds[0][:min_len]
            correlations = []
            for pred in all_preds[1:]:
                if len(pred) >= min_len:
                    corr = np.corrcoef(first_pred, pred[:min_len])[0, 1]
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)
            avg_correlation = np.mean(correlations) if correlations else 0
            return min(1.0, max(0.0, avg_correlation))
        except:
            return 0.5

    def _format_predictions_to_dataframe(self, predictions: Dict, confidence_intervals: Dict,
                                         historical_data: pd.DataFrame, days_ahead: int) -> pd.DataFrame:
        try:
            last_date = historical_data.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=days_ahead,
                freq='B'
            )
            result_df = pd.DataFrame(index=future_dates)
            ensemble_pred = predictions.get('ensemble')
            if ensemble_pred is not None and len(ensemble_pred) == days_ahead:
                result_df['Predicted_Close'] = ensemble_pred
            if confidence_intervals:
                lower = confidence_intervals.get('lower')
                upper = confidence_intervals.get('upper')
                if lower is not None and upper is not None and len(lower) == days_ahead and len(upper) == days_ahead:
                    result_df['CI_Lower'] = lower
                    result_df['CI_Upper'] = upper
            if 'Close' in historical_data.columns and 'Predicted_Close' in result_df.columns:
                current_price = historical_data['Close'].iloc[-1]
                result_df['Expected_Return'] = (result_df['Predicted_Close'] - current_price) / current_price
            return result_df
        except Exception as e:
            logger.error(f"Erreur formatage prédictions: {e}")
            return pd.DataFrame()

    def _get_empty_prediction_result(self) -> Dict:
        return {
            'predictions': pd.DataFrame(),
            'signals': {'overall_signal': 'NEUTRAL', 'signals': []},
            'metrics': {},
            'scenarios': None,
            'error': 'Prédiction impossible'
        }

    def _handle_prediction_error(self, error: Exception) -> Dict:
        return {
            'predictions': pd.DataFrame(),
            'signals': {'overall_signal': 'ERROR', 'signals': []},
            'metrics': {'error': str(error)},
            'scenarios': None,
            'error': str(error)
        }

    def _save_prediction_history(self, predictions_df: pd.DataFrame, metrics: Dict):
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
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
        except Exception as e:
            logger.error(f"Erreur sauvegarde historique _save_prediction_history: {e}")

    def get_prediction_history(self) -> List[Dict]:
        return self.prediction_history

    def _get_technical_summary(self) -> Dict[str, Any]:
            """Extrait et formate les indicateurs techniques pour le rapport final."""
            try:
                if self.features is None or self.features.empty:
                    logger.warning("Features vides pour le résumé technique.")
                    return {}

                last_row = self.features.iloc[-1]
                # Mapping pour potentiellement ajouter des descriptions ou arrondis spécifiques
                key_indicators = {
                    'RSI': 'Relative Strength Index',
                    'MACD': 'Moving Average Convergence Divergence',
                    'SMA_20': 'Simple MA 20',
                    'SMA_50': 'Simple MA 50',
                    'SMA_200': 'Simple MA 200',
                    'ATR': 'Average True Range',
                    'BB_Width': 'Bollinger Band Width'
                }
            
                return {
                    label: round(float(last_row[col]), 4) 
                    for col, label in key_indicators.items() 
                    if col in last_row
                }
            except Exception as e:
                logger.error(f"⚠️ Erreur résumé technique: {e}")
                return {}

    def _get_market_context(self) -> Dict[str, Any]:
        """Récupère le contexte macro-économique avec gestion des erreurs par service."""
        context = {}
        try:
            # On utilise les instances déjà existantes si possible pour économiser les ressources
            macro = getattr(self, 'macro_extractor', MacroDataExtractor())
            
            # Appels isolés pour éviter qu'une erreur sur un service ne bloque tout le contexte
            try:
                context["market_indicators"] = self.market_data_extractor.get_market_indicators()
            except Exception: context["market_indicators"] = "Indisponible"

            try:
                context["economic_indicators"] = macro.get_economic_indicators("US")
            except Exception: context["economic_indicators"] = {}

            try:
                context["commodities"] = macro.get_commodity_prices()
            except Exception: context["commodities"] = {}

            return context
        except Exception as e:
            logger.error(f"⚠️ Erreur globale contexte marché: {e}")
            return {}

# --- POINT D'ENTRÉE ---

def main():
    """Script de test principal avec affichage formaté."""
    print("\n" + "="*60)
    print("🚀 STOCK PREDICTOR PRO - Analyse Multidimensionnelle")
    print("="*60)

    symbol = "AAPL"
    
    try:
        # Initialisation avec balance de test
        predictor = StockPredictor(symbol, period="3y", account_balance=50000.0)
        
        print(f"\n[1/3] 📡 Récupération des données et entraînement pour {symbol}...")
        results = predictor.analyze_stock_advanced()

        if 'error' in results:
            print(f"❌ Échec de l'analyse: {results['error']}")
            return

        # --- AFFICHAGE DES RÉSULTATS ---
        print(f"\n[2/3] 📊 Synthèse du Signal")
        print("-" * 30)
        print(f"🔹 Prix actuel      : ${results['current_price']:.2f}")
        print(f"🔹 Score de Trading : {results['trading_score']}/10")
        print(f"🔹 Recommandation   : {results['recommendation'].upper()}")

        print(f"\n[3/3] 🛡️ Gestion du Risque (Horizon Court Terme)")
        risk = results.get('risk_metrics', {})
        h_1d = risk.get('horizons', {}).get('1d', {})
        
        if h_1d:
            print(f"📉 Stop-Loss suggéré : ${h_1d.get('stop_loss')}")
            print(f"📈 Take-Profit cible : ${h_1d.get('take_profit')}")
            print(f"💰 Taille position   : {h_1d.get('position_sizing')} unités")
        
        print("\n" + "="*60)
        print("✅ Rapport généré avec succès.")

    except KeyboardInterrupt:
        print("\n🛑 Analyse interrompue par l'utilisateur.")
    except Exception as e:
        print(f"\n❌ Erreur critique : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
