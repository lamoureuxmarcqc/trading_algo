"""
Module principal pour les prédictions et trading d'actions
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import pickle
import numpy as np
import pandas as pd
import gc
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
# Modules internes
from trading_algo.data.data_extraction import StockDataExtractor, get_stock_overview, MacroDataExtractor
from trading_algo.models.base_model import ImprovedLSTMPredictorMultiOutput

# Do not configure logging at module import time.
logger = logging.getLogger(__name__)
tf.get_logger().setLevel('ERROR')

class StockModelTrain:
    """
    Classe principale pour les prédictions d'actions avec IA
    Intègre les modules data_extraction et dashboard
    """
    
    def __init__(self, symbol: str, period: str = "3y"):
        self.symbol = symbol
        self.period = period
        self.model = None
        self.data = None
        self.features = None
        self.feature_columns = None
        self.targets = None
        self.target_columns = None
        self.current_price = 0.0
        
        # Extracteurs de données
        self.data_extractor = StockDataExtractor(symbol)
        self.macro_extractor = MacroDataExtractor()
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Prédictions
        self.predictions_1d = None
        self.predictions_30d = None
        self.predictions_90d = None
        
        # Résultats d'analyse
        self.analysis_results = {}
        self.trading_score = 5.0
        self.recommendation = "NEUTRE"
        
        # Configuration
        self.sequence_length = 60 # valeur par défaut, sera mise à jour pendant l'entraînement
        self.lookback_days = 60 # alias pour cohérence
        self.batch_size = 32
        self.patience_early_stopping = 20
        
        logger.info(f"StockModelTrain initialisé pour {symbol}")
    
    def _check_gpu(self):
        """Vérifie et configure les GPUs disponibles"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU détecté: {len(gpus)} disponible(s)")
                return True
            else:
                logger.info("Aucun GPU détecté, utilisation du CPU")
                return False
        except Exception as e:
            logger.warning(f"Erreur configuration GPU: {e}")
            return False
    
    def fetch_data(self, forecast_horizon: int = 1) -> bool:
        """
        Récupère et prépare les données pour l'analyse
        """
        try:
            logger.info(f"Récupération des données pour {self.symbol}")
            
            all_data = self.data_extractor.get_all_data(
                symbol=self.symbol,
                period=self.period
            )
            
            if all_data is None or not isinstance(all_data, dict) or 'historical' not in all_data:
                logger.error("Données historiques manquantes")
                return False
            
            historical_data = all_data['historical']
            if historical_data is None or historical_data.empty:
                logger.error("Données historiques vides")
                return False
            
            self.data = historical_data
            
            if 'technical' in all_data and all_data['technical'] is not None and not all_data['technical'].empty:
                self.features = all_data['technical']
            else:
                self.features = self.data_extractor.calculate_technical_indicators(self.data)
            
            if 'targets' in all_data and all_data['targets'] is not None and not all_data['targets'].empty:
                self.targets = all_data['targets']
            else:
                self.targets = self.data_extractor.create_target_columns(
                    self.features,
                    forecast_days=[1, 5, 10, 20, 30, 90]
                )
            
            if self.targets is None or self.targets.empty:
                logger.error("Échec de la création des cibles")
                return False
            
            if not self.data.empty and 'Close' in self.data.columns:
                self.current_price = float(self.data['Close'].iloc[-1])
            
            if self.features is not None and not self.features.empty:
                self.feature_scaler.fit(self.features)
            
            if self.targets is not None and not self.targets.empty:
                self.target_scaler.fit(self.targets.dropna())
            
            logger.info(f"Données récupérées: {len(self.data)} périodes")
            logger.info(f"Features: {self.features.shape}")
            logger.info(f"Cibles: {self.targets.shape}")
            logger.info(f"Prix actuel: ${self.current_price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur fetch_data: {e}", exc_info=True)
            return False
    
    def _check_data_alignment(self):
        """Vérifie l'alignement entre data et features"""
        try:
            if self.data is None or self.features is None:
                logger.warning("Données ou features manquantes")
                return False
            
            data_dates = set(self.data.index)
            feature_dates = set(self.features.index)
            common_dates = data_dates.intersection(feature_dates)
            
            logger.info(f"Dates data: {len(data_dates)}, Dates features: {len(feature_dates)}, "
                        f"Dates communes: {len(common_dates)}")
            
            if len(common_dates) < min(len(data_dates), len(feature_dates)) * 0.8:
                logger.warning(f"Alignement faible: {len(common_dates)/len(data_dates)*100:.1f}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur _check_data_alignment: {e}")
            return False
    
    def prepare_training_data(self, lookback_days: int = 30, train_split: float = 0.8) -> Tuple:
        """
        Prépare les données pour l'entraînement du modèle
        """
        try:
            if self.features is None or self.targets is None:
                logger.error("Données non disponibles")
                return None, None, None, None
            
            self._check_data_alignment()
            
            X_df = self.features.copy()
            y_df = self.targets.copy()
            
            common_index = X_df.index.intersection(y_df.index)
            X_df = X_df.loc[common_index]
            y_df = y_df.loc[common_index]
            
            X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            y_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            X_df = X_df.ffill().bfill()
            y_df = y_df.ffill().bfill()
            
            split_idx = int(len(X_df) * train_split)
            
            X_train_df = X_df.iloc[:split_idx]
            X_test_df = X_df.iloc[split_idx:]
            y_train_df = y_df.iloc[:split_idx]
            y_test_df = y_df.iloc[split_idx:]
            
            # Après avoir préparé X_train_df (avant scaling), vers la fin de la méthode train()
            self.feature_columns = X_train_df.columns.tolist()
            logger.info(f"Colonnes des features sauvegardées: {len(self.feature_columns)}")
            # Après avoir préparé y_train_df, avant le scaling
            self.target_columns = y_train_df.columns.tolist()
            logger.info(f"Colonnes cibles sauvegardées: {len(self.target_columns)}")
            if len(X_train_df) < lookback_days + 10:
                logger.warning("Données insuffisantes pour l'entraînement")
                return None, None, None, None
            
            X_train = self.feature_scaler.transform(X_train_df.values)
            X_test = self.feature_scaler.transform(X_test_df.values)
            
            y_train = self.target_scaler.transform(y_train_df.values)
            y_test = self.target_scaler.transform(y_test_df.values)
            
            X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, lookback_days)
            X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, lookback_days)
            
            logger.info(f"Séquences d'entraînement: {X_train_seq.shape}")
            logger.info(f"Séquences de test: {X_test_seq.shape}")
            
            return X_train_seq, y_train_seq, X_test_seq, y_test_seq
            
        except Exception as e:
            logger.error(f"Erreur prepare_training_data: {e}")
            return None, None, None, None
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, lookback_days: int) -> Tuple:
        """Crée des séquences pour les modèles LSTM"""
        if len(X) <= lookback_days:
            return np.array([]).reshape(0, lookback_days, X.shape[1]), np.array([]).reshape(0, y.shape[1])
        
        X_seq = np.zeros((len(X) - lookback_days, lookback_days, X.shape[1]))
        y_seq = np.zeros((len(X) - lookback_days, y.shape[1]))
        
        for i in range(lookback_days, len(X)):
            X_seq[i - lookback_days] = X[i - lookback_days:i]
            y_seq[i - lookback_days] = y[i]
        
        return X_seq, y_seq

    def export_training_data(self, filename: str = None) -> str:
        """
        Exporte les données préparées pour l'entraînement dans un fichier Excel.
        Neutralise toutes les timezones (index + colonnes datetime) et gère les types non supportés.
        """
        try:
            if self.features is None or self.targets is None:
                logger.error("Pas de données à exporter. Exécutez fetch_data() d'abord.")
                return None

            # ============================================================
            # 1) Neutraliser les timezones AVANT l'intersection
            # ============================================================
            feat_idx = self.features.index
            targ_idx = self.targets.index

            if getattr(feat_idx, "tz", None) is not None:
                feat_idx = feat_idx.tz_localize(None)

            if getattr(targ_idx, "tz", None) is not None:
                targ_idx = targ_idx.tz_localize(None)

            # Appliquer les index neutralisés aux DataFrames
            X_df = self.features.copy()
            y_df = self.targets.copy()

            X_df.index = feat_idx
            y_df.index = targ_idx

            # Intersection cohérente (timezone-naive)
            common_idx = feat_idx.intersection(targ_idx)

            # Filtrer les lignes communes
            X_df = X_df.loc[common_idx].copy()
            y_df = y_df.loc[common_idx].copy()

            # ============================================================
            # 2) Neutralisation robuste des timezones dans les colonnes
            # ============================================================
            def neutralize_timezones_in_df(df):
                for col in df.columns:
                    # Forcer conversion en datetime si possible
                    try:
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                    except Exception:
                        pass

                    # Neutraliser timezone si datetime
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        try:
                            if df[col].dt.tz is not None:
                                df[col] = df[col].dt.tz_localize(None)
                        except Exception:
                            df[col] = df[col].apply(
                                lambda x: x.tz_localize(None) if hasattr(x, "tz_localize") else x
                            )
                return df

            X_df = neutralize_timezones_in_df(X_df)
            y_df = neutralize_timezones_in_df(y_df)

            # ============================================================
            # 3) Nettoyage des types non supportés par Excel (sans applymap)
            # ============================================================
            def sanitize_for_excel(df):
                return df.apply(
                    lambda col: col.map(
                        lambda x: str(x) if isinstance(x, (list, dict, tuple, set)) else x
                    )
                )

            X_df = sanitize_for_excel(X_df)
            y_df = sanitize_for_excel(y_df)

            # ============================================================
            # 4) Définir le nom du fichier
            # ============================================================
            if filename is None:
                filename = f"{self.symbol}_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            # ============================================================
            # 5) Export Excel
            # ============================================================
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:

                # Feuille features
                X_df.to_excel(writer, sheet_name='features')

                # Feuille targets
                y_df.to_excel(writer, sheet_name='targets')

                # Feuille statistiques descriptives
                stats_list = []
                for col in X_df.columns:
                    stats_list.append({
                        'type': 'feature',
                        'colonne': col,
                        'moyenne': X_df[col].mean(),
                        'ecart_type': X_df[col].std(),
                        'min': X_df[col].min(),
                        'max': X_df[col].max(),
                        'nb_nan': X_df[col].isna().sum(),
                        'pct_nan': X_df[col].isna().mean() * 100
                    })
                for col in y_df.columns:
                    stats_list.append({
                        'type': 'target',
                        'colonne': col,
                        'moyenne': y_df[col].mean(),
                        'ecart_type': y_df[col].std(),
                        'min': y_df[col].min(),
                        'max': y_df[col].max(),
                        'nb_nan': y_df[col].isna().sum(),
                        'pct_nan': y_df[col].isna().mean() * 100
                    })

                stats_df = pd.DataFrame(stats_list)
                stats_df.to_excel(writer, sheet_name='stats', index=False)

                # ============================================================
                # 6) Feuille info (dates timezone-naive garanties)
                # ============================================================
                start_date = common_idx.min()
                end_date = common_idx.max()

                info_data = {
                    'symbol': [self.symbol],
                    'period': [self.period],
                    'start_date': [start_date],
                    'end_date': [end_date],
                    'n_rows': [len(common_idx)],
                    'n_features': [X_df.shape[1]],
                    'n_targets': [y_df.shape[1]],
                    'feature_columns': [', '.join(X_df.columns[:5]) + '...'],
                    'target_columns': [', '.join(y_df.columns[:5]) + '...']
                }

                pd.DataFrame(info_data).to_excel(writer, sheet_name='info', index=False)

            logger.info(f"✅ Données exportées dans {filename}")
            return filename

        except Exception as e:
            logger.error(f"Erreur lors de l'exportation des données en excel : {e}", exc_info=True)
            return None


        
    def train(self, lookback_days: int = 30, epochs: int = 50, batch_size: int = 32) -> bool:
        """
        Entraîne le modèle de prédiction avec améliorations
        """
        try:
            # Gestion mémoire
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Sauvegarde du lookback_days pour l'utiliser en prédiction
            self.lookback_days = lookback_days
            self.sequence_length = lookback_days # pour compatibilité
            
            # Configuration des dossiers
            model_dir = f"models_saved/{self.symbol}"
            checkpoints_dir = f"checkpoints/{self.symbol}"
            log_dir = f"logs/{self.symbol}"
            
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(checkpoints_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            
            self._check_gpu()
            
            X_train, y_train, X_test, y_test = self.prepare_training_data(lookback_days)
            
            if X_train is None or X_train.shape[0] == 0:
                logger.error("Données insuffisantes pour l'entraînement")
                return False
            
            if X_train.shape[0] < batch_size * 2:
                new_batch_size = max(1, X_train.shape[0] // 4)
                logger.warning(f"Batch size ajusté de {batch_size} à {new_batch_size}")
                batch_size = new_batch_size
            
            n_outputs = y_train.shape[1]
            logger.info(f"Données d'entraînement: {X_train.shape[0]} échantillons, {n_outputs} cibles")
            logger.info(f"Données de test: {X_test.shape[0]} échantillons")
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            if X_train.shape[0] < 500:
                lstm_units1, lstm_units2, lstm_units3 = 32, 16, 8
                logger.info("Architecture légère sélectionnée (petit dataset 32, 16, 8)")
            else:
                lstm_units1, lstm_units2, lstm_units3 = 128, 64, 32
                logger.info("Architecture sélectionnée (128, 64, 32)")
            
            self.model = ImprovedLSTMPredictorMultiOutput(
                lstm_units1=lstm_units1,
                lstm_units2=lstm_units2,
                lstm_units3=lstm_units3,
                dense_units=64,
                dropout_rate=0.3,
                recurrent_dropout=0.2,
                l2_reg=0.001,
                n_outputs=n_outputs
            )
            
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                clipnorm=1.0
            )
            
            self.model.compile(
                optimizer=optimizer,
                loss="huber",
                metrics=["mae", "mse"]
            )
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{checkpoints_dir}/best_model_{timestamp}.keras",
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience_early_stopping,
                    min_delta=0.0001,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=8,
                    min_lr=1e-7,
                    verbose=1
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=f"{log_dir}/{timestamp}",
                    histogram_freq=1
                )
            ]
            
            logger.info(f"Début de l'entraînement pour {epochs} epochs")
            start_time = datetime.now()
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                shuffle=False,
                verbose=1
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Temps d'entraînement: {training_time:.2f} secondes")
            # Après l'entraînement, vers la fin de la méthode train()
            if X_test is not None and y_test is not None:
                self.validate_on_test(X_test, y_test)
            
            train_metrics = self.model.evaluate(X_train, y_train, verbose=0, return_dict=True)
            val_metrics = self.model.evaluate(X_test, y_test, verbose=0, return_dict=True)
            
            model_path = f"{model_dir}/{self.symbol}_model_{timestamp}.keras"
            self.model.save(model_path)
            logger.info(f"Modèle sauvegardé: {model_path}")
            
            feature_scaler_path = f"{model_dir}/{self.symbol}_feature_scaler_{timestamp}.pkl"
            with open(feature_scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            
            target_scaler_path = f"{model_dir}/{self.symbol}_target_scaler_{timestamp}.pkl"
            with open(target_scaler_path, 'wb') as f:
                pickle.dump(self.target_scaler, f)
            
            metrics = {
                'symbol': self.symbol,
                'timestamp': timestamp,
                'training_time_seconds': training_time,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_loss': float(train_metrics['loss']),
                'train_mae': float(train_metrics['mae']),
                'val_loss': float(val_metrics['loss']),
                'val_mae': float(val_metrics['mae']),
                'input_shape': input_shape,
                'n_outputs': n_outputs,
                'lookback_days': lookback_days,
                'batch_size': batch_size,
                'architecture': f"LSTM_{lstm_units1}_{lstm_units2}",
                'model_version': '1.0.0'
            }
            
            self._save_metrics(metrics, model_dir)
            self._plot_training_history(history, model_dir, timestamp)
            del X_train, y_train, X_test, y_test
            gc.collect()
            
            logger.info("Entraînement terminé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
            return False
    
    def validate_on_test(self, X_test_seq, y_test_seq):
        """
        Compare les prédictions du modèle sur l'ensemble de test avec les valeurs réelles.
        X_test_seq : séquences d'entrée pour le test (shape: nb_seq, lookback, n_features)
        y_test_seq : cibles correspondantes (shape: nb_seq, n_targets)
        """
        if self.model is None:
            logger.error("Modèle non entraîné")
            return
        
        y_pred_scaled = self.model.predict(X_test_seq, verbose=0)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        y_true = self.target_scaler.inverse_transform(y_test_seq) # y_test_seq est déjà scaled
        
        logger.info("=== Validation sur données de test ===")
        for i, col in enumerate(self.target_columns):
            if 'Target_Close' in col: # On se concentre sur les prix de clôture futurs
                mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
                logger.info(f"MAE pour {col} : {mae:.2f}")
                # Afficher les 3 premiers exemples
                for j in range(min(3, len(y_pred))):
                    logger.info(f" Exemple {j}: prédit={y_pred[j, i]:.2f}, réel={y_true[j, i]:.2f}")
    
    def _save_metrics(self, metrics: Dict[str, Any], save_dir: str):
        """Sauvegarde les métriques d'entraînement"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{save_dir}/metrics_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Métriques sauvegardées: {filename}")
        except Exception as e:
            logger.error(f"Erreur save_metrics: {e}")
    
    def _plot_training_history(self, history, model_dir, timestamp):
        """Génère et sauvegarde des graphiques d'entraînement"""
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(history.history['loss'], label='Train Loss')
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
            axes[0].set_title('Model Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            axes[1].plot(history.history['mae'], label='Train MAE')
            axes[1].plot(history.history['val_mae'], label='Validation MAE')
            axes[1].set_title('Model MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            plot_path = f"{model_dir}/training_history_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Graphique d'entraînement sauvegardé: {plot_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique: {e}")
    
    def analyze_model_stock(self) -> Dict[str, Any]:
        """
        Analyse complète de l'action
        """
        try:
            logger.info(f"Analyse de {self.symbol}")
            
            if not self.fetch_data():
                return {"error": "Échec de la récupération des données", "symbol": self.symbol}
            
            logger.info("Entraînement du modèle IA...")
            training_success = self.train(epochs=30, lookback_days=60)
            
            if not training_success:
                return {
                    "error": "Échec de l'entraînement du modèle",
                    "symbol": self.symbol,
                    "current_price": self.current_price
                }
            
            logger.info("Génération des prédictions...")
            predictions = self.generate_predictions()
            
            self.trading_score = self.calculate_trading_score(predictions)
            self.recommendation = self.generate_recommendation(self.trading_score)
            
            self.analysis_results = {
                "symbol": self.symbol,
                "current_price": self.current_price,
                "trading_score": self.trading_score,
                "recommendation": self.recommendation,
                "predictions": predictions,
                "analysis_date": datetime.now().isoformat(),
                "period": self.period
            }
            
            logger.info(f"Analyse terminée pour {self.symbol}")
            logger.info(f"Score: {self.trading_score}/10 - {self.recommendation}")
            
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Erreur analyze_model_stock: {e}", exc_info=True)
            return {"error": str(e), "symbol": self.symbol}
    
    def generate_predictions(self) -> Dict[str, Any]:
        try:
            if self.model is None or self.features is None:
                logger.error("Modèle ou features non disponibles")
                return {}
        
            if not hasattr(self, 'feature_columns'):
                logger.error("Aucune colonne de features sauvegardée. Réentraînez le modèle.")
                return {}
        
            missing = set(self.feature_columns) - set(self.features.columns)
            if missing:
                logger.error(f"Colonnes manquantes dans features: {missing}")
                return {}
        
            features_aligned = self.features[self.feature_columns]
        
            if len(features_aligned) < self.lookback_days:
                logger.error(f"Pas assez de données : besoin de {self.lookback_days} jours, disponible {len(features_aligned)}")
                return {}
            excel_file = self.export_training_data()
            recent_data = features_aligned.iloc[-self.lookback_days:]
            logger.info(f"Dernières dates: {recent_data.index[-1]}")
            logger.info(f"Valeurs manquantes dans recent_data: {recent_data.isna().sum().sum()}")
            scaled_data = self.feature_scaler.transform(recent_data)
            X_pred = scaled_data.reshape(1, self.lookback_days, -1)
        
            y_pred_scaled = self.model.predict(X_pred, verbose=0)
        
            if not hasattr(self, 'target_columns'):
                logger.error("target_columns non définies. Utilisation de l'ordre par défaut (risqué).")
                y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
                return {
                    "1d": float(y_pred[0, 0]) if y_pred.shape[1] > 0 else None,
                    "5d": float(y_pred[0, 1]) if y_pred.shape[1] > 1 else None,
                    "10d": float(y_pred[0, 2]) if y_pred.shape[1] > 2 else None,
                    "20d": float(y_pred[0, 3]) if y_pred.shape[1] > 3 else None,
                    "30d": float(y_pred[0, 4]) if y_pred.shape[1] > 4 else None,
                    "90d": float(y_pred[0, 5]) if y_pred.shape[1] > 5 else None
                }
        
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
            pred_dict = {col: y_pred[0, i] for i, col in enumerate(self.target_columns)}
        
            horizons = [1, 5, 10, 20, 30, 90]
            predictions = {}
            for days in horizons:
                close_col = f'Target_Close_{days}d'
                predictions[f'{days}d'] = float(pred_dict[close_col]) if close_col in pred_dict else None
        
            logger.info(f"Prédictions extraites: {predictions}")
            return predictions
        
        except Exception as e:
            logger.error(f"Erreur generate_predictions: {e}")
            return {}
    
    def calculate_trading_score(self, predictions: Dict[str, Any]) -> float:
        """
        Calcule un score de trading basé sur les prédictions
        """
        try:
            score = 5.0
            if not predictions:
                return score
            
            returns = {}
            for horizon, pred in predictions.items():
                if pred is not None and self.current_price > 0:
                    returns[horizon] = (pred - self.current_price) / self.current_price * 100
            
            weights = {"1d": 0.1, "5d": 0.15, "10d": 0.2, "30d": 0.3, "90d": 0.25}
            weighted_return = 0
            total_weight = 0
            
            for horizon, weight in weights.items():
                if horizon in returns and returns[horizon] is not None:
                    weighted_return += returns[horizon] * weight
                    total_weight += weight
            
            if total_weight > 0:
                avg_return = weighted_return / total_weight
                score = 5 + (avg_return / 2)
                score = max(0, min(10, score))
            
            return round(score, 2)
            
        except Exception as e:
            logger.error(f"Erreur calculate_trading_score: {e}")
            return 5.0
    
    def generate_recommendation(self, score: float) -> str:
        """
        Génère une recommandation basée sur le score
        """
        if score >= 7.5:
            return "ACHAT FORT"
        elif score >= 6:
            return "ACHAT"
        elif score >= 4:
            return "NEUTRE"
        elif score >= 2.5:
            return "VENTE"
        else:
            return "VENTE FORTE"
    
    def create_dashboard(self):
        """
        Crée un tableau de bord interactif (à implémenter avec le module dashboard)
        """
        try:
            if not self.analysis_results:
                self.analyze_model_stock()
            
            logger.info(f"\n📊 DASHBOARD POUR {self.symbol}")
            logger.info("=" * 60)
            logger.info(f"Prix actuel: ${self.current_price:.2f}")
            logger.info(f"Score de trading: {self.trading_score}/10")
            logger.info(f"Recommandation: {self.recommendation}")
            logger.info("\nPrédictions:")
            
            predictions = self.analysis_results.get('predictions', {})
            for horizon, pred in predictions.items():
                if pred is not None:
                    change_pct = ((pred - self.current_price) / self.current_price * 100)
                    logger.info(f" {horizon}: ${pred:.2f} ({change_pct:+.2f}%)")
            
            logger.info("\nNote: Le dashboard interactif sera disponible avec le module dashboard.py")
            
        except Exception as e:
            logger.error(f"Erreur stockmodeltrain.create_dashboard: {e}", exc_info=True)


def main():
    """Fonction principale d'exécution dans stockmodeltrain"""
    logger.info("🚀 PRÉDICTEUR D'ACTIONS AVEC IA")
    logger.info("=" * 60)
    
    popular_stocks = {
        '1': 'AAPL', '2': 'TSLA', '3': 'MSFT', '4': 'GOOGL',
        '5': 'AMZN', '6': 'META', '7': 'NVDA', '8': 'BTC-USD', '9': 'SPY',
    }
    
    logger.info("\n📈 ACTIONS POPULAIRES:")
    for key, value in popular_stocks.items():
        logger.info(f" {key}. {value}")
    logger.info(" 0. Entrer un symbole personnalisé")
    
    choice = input("\nChoisissez une action (1-9) ou 0 pour personnalisé: ").strip()
    
    if choice == '0':
        symbol = input("Entrez le symbole de l'action (ex: AAPL): ").strip().upper()
    elif choice in popular_stocks:
        symbol = popular_stocks[choice]
    else:
        logger.warning("Choix invalide, utilisation de AAPL par défaut")
        symbol = 'AAPL'
    
    logger.info("\n⏰ PÉRIODE D'ANALYSE:")
    logger.info(" 1. 1 an")
    logger.info(" 2. 3 ans")
    logger.info(" 3. 5 ans (recommandé)")
    logger.info(" 4. 10 ans")
    
    period_choice = input("Choisissez la période (1-4): ").strip()
    periods = {'1': '1y', '2': '3y', '3': '5y', '4': '10y'}
    period = periods.get(period_choice, '5y')
    
    logger.info("\n⚙️ OPTIONS D'ANALYSE:")
    logger.info(" 1. Analyse complète avec dashboard")
    logger.info(" 2. Analyse rapide (sans dashboard)")
    
    option_choice = input("Choisissez l'option (1-2): ").strip()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Lancement de l'analyse pour {symbol} ({period})...")
    logger.info(f"{'='*60}")
    
    start_time = datetime.now()
    
    try:
        predictor = StockModelTrain(symbol, period)
        results = predictor.analyze_model_stock()
        
        if 'error' not in results:
            logger.info(f"\n📊 RÉSULTATS POUR {symbol}:")
            logger.info(f" Prix actuel: ${results['current_price']:.2f}")
            logger.info(f" Score de trading: {results['trading_score']:.1f}/10")
            logger.info(f" Recommandation: {results['recommendation']}")
            
            predictions = results['predictions']
            logger.info(f"\n Prédictions de prix:")
            for horizon, pred in predictions.items():
                if pred is not None:
                    change_pct = ((pred - results['current_price']) / results['current_price'] * 100)
                    logger.info(f" {horizon}: ${pred:.2f} ({change_pct:+.2f}%)")
            
            if option_choice == '1':
                logger.info("\n📈 Création du dashboard...")
                predictor.create_dashboard()
        else:
            logger.error(f"❌ Erreur: {results['error']}")
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse: {e}", exc_info=True)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"\n⏱️ Analyse terminée en {duration:.1f} secondes")
    logger.info("=" * 60)

# When executed as a script, initialize logging centrally first.
if __name__ == "__main__":
    try:
        from trading_algo.logging_config import init_logging
        init_logging(level=os.getenv("LOG_LEVEL", None), logfile=os.getenv("LOG_FILE", None))
    except Exception:
        # fallback to basicConfig only for direct script execution environment
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Execute main CLI behavior
    main()