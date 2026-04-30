"""
Module principal pour les prédictions et trading d'actions - Version compatible avec la nouvelle architecture data_extraction
Classes spécialisées : StockDataExtractor, FundamentalExtractor, MacroDataExtractor, etc.
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
from sklearn.preprocessing import StandardScaler

from trading_algo.data.data_extraction import (
    StockDataExtractor,
    MacroDataExtractor,
    FundamentalExtractor,
    SentimentExtractor,
    fetch_stock_data,
    get_stock_overview
)
from trading_algo.models.base_model import ImprovedLSTMPredictorMultiOutput

logger = logging.getLogger(__name__)
tf.get_logger().setLevel('ERROR')


class StockModelTrain:
    """
    Classe principale pour les prédictions d'actions avec IA
    Version optimisée, compatible avec la nouvelle architecture data_extraction.
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

        # Extracteurs (utiliser les nouvelles classes)
        self.data_extractor = StockDataExtractor(symbol)
        self.macro_extractor = MacroDataExtractor()
        self.fundamental_extractor = FundamentalExtractor()
        self.sentiment_extractor = SentimentExtractor()

        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # Résultats
        self.analysis_results = {}
        self.trading_score = 5.0
        self.recommendation = "NEUTRE"

        # Hyperparamètres améliorés
        self.lookback_days = 90          # fenêtre augmentée
        self.sequence_length = 90
        self.batch_size = 32
        self.patience_early_stopping = 20

        logger.info(f"StockModelTrain optimisé pour {symbol} (lookback={self.lookback_days})")

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
        """Récupère et prépare les données pour l'analyse en utilisant fetch_stock_data (refactorisé)."""
        try:
            logger.info(f"Récupération des données pour {self.symbol}")
            all_data = fetch_stock_data(symbol=self.symbol, period=self.period, include_technicals=True)
            if all_data is None or not isinstance(all_data, dict) or 'historical' not in all_data:
                logger.error("Données historiques manquantes")
                return False

            historical_data = all_data['historical']
            if historical_data is None or historical_data.empty:
                logger.error("Données historiques vides")
                return False

            self.data = historical_data

            # Utiliser les indicateurs techniques pré-calculés si disponibles
            if 'technical' in all_data and all_data['technical'] is not None and not all_data['technical'].empty:
                self.features = all_data['technical']
            else:
                self.features = self.data_extractor.calculate_technical_indicators(self.data)

            # Utiliser les cibles pré-calculées si disponibles
            if 'targets' in all_data and all_data['targets'] is not None and not all_data['targets'].empty:
                self.targets = all_data['targets']
            else:
                self.targets = self.data_extractor.create_target_columns(
                    self.features, forecast_days=[1, 5, 10, 20, 30, 90]
                )

            if self.targets is None or self.targets.empty:
                logger.error("Échec de la création des cibles")
                return False

            self.current_price = float(self.data['Close'].iloc[-1])

            # Transformation logarithmique des cibles (prix -> log prix)
            target_cols = [c for c in self.targets.columns if 'Target_Close' in c]
            for col in target_cols:
                self.targets[col] = np.log(self.targets[col])

            # Sauvegarde des colonnes pour référence ultérieure
            if self.features is not None and not self.features.empty:
                self.feature_columns = self.features.columns.tolist()
            if self.targets is not None and not self.targets.empty:
                self.target_columns = self.targets.columns.tolist()

            logger.info(f"Données récupérées: {len(self.data)} périodes")
            logger.info(f"Features: {self.features.shape if self.features is not None else 'None'}")
            logger.info(f"Cibles (log): {self.targets.shape if self.targets is not None else 'None'}")
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

    def prepare_training_data(self, lookback_days: int = None, train_split: float = 0.8) -> Tuple:
        """
        Prépare les données pour l'entraînement.
        Utilise self.lookback_days par défaut.
        """
        if lookback_days is None:
            lookback_days = self.lookback_days
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

            # Fit des scalers UNIQUEMENT sur l'entraînement
            self.feature_scaler.fit(X_train_df)
            self.target_scaler.fit(y_train_df)

            # Sauvegarde des colonnes
            self.feature_columns = X_train_df.columns.tolist()
            self.target_columns = y_train_df.columns.tolist()
            logger.info(f"Colonnes des features sauvegardées: {len(self.feature_columns)}")
            logger.info(f"Colonnes cibles sauvegardées: {len(self.target_columns)}")

            if len(X_train_df) < lookback_days + 10:
                logger.warning("Données insuffisantes pour l'entraînement")
                return None, None, None, None

            # Transformation avec les scalers fittés sur l'entraînement
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
            logger.error(f"Erreur prepare_training_data: {e}", exc_info=True)
            return None, None, None, None

    def _create_sequences(self, X: np.ndarray, y: np.ndarray, lookback_days: int) -> Tuple:
        """Crée des séquences pour les modèles LSTM (dtype float32)"""
        if len(X) <= lookback_days:
            return (np.empty((0, lookback_days, X.shape[1]), dtype=np.float32),
                    np.empty((0, y.shape[1]), dtype=np.float32))
        X_seq = np.zeros((len(X) - lookback_days, lookback_days, X.shape[1]), dtype=np.float32)
        y_seq = np.zeros((len(X) - lookback_days, y.shape[1]), dtype=np.float32)
        for i in range(lookback_days, len(X)):
            X_seq[i - lookback_days] = X[i - lookback_days:i].astype(np.float32)
            y_seq[i - lookback_days] = y[i].astype(np.float32)
        return X_seq, y_seq

    def export_training_data(self, filename: str = None) -> str:
        """
        Exporte les données préparées pour l'entraînement dans un fichier Excel.
        Neutralise les timezones et gère les types non supportés.
        """
        try:
            if self.features is None or self.targets is None:
                logger.error("Pas de données à exporter. Exécutez fetch_data() d'abord.")
                return None

            # Neutralisation timezone
            feat_idx = self.features.index
            targ_idx = self.targets.index
            if getattr(feat_idx, "tz", None) is not None:
                feat_idx = feat_idx.tz_localize(None)
            if getattr(targ_idx, "tz", None) is not None:
                targ_idx = targ_idx.tz_localize(None)

            X_df = self.features.copy()
            y_df = self.targets.copy()
            X_df.index = feat_idx
            y_df.index = targ_idx
            common_idx = feat_idx.intersection(targ_idx)
            X_df = X_df.loc[common_idx].copy()
            y_df = y_df.loc[common_idx].copy()

            # Conversion datetime
            for col in X_df.columns:
                try:
                    X_df[col] = pd.to_datetime(X_df[col], errors='ignore')
                except:
                    pass
            for col in y_df.columns:
                try:
                    y_df[col] = pd.to_datetime(y_df[col], errors='ignore')
                except:
                    pass

            # Nettoyage Excel
            def sanitize(df):
                return df.apply(lambda col: col.map(lambda x: str(x) if isinstance(x, (list, dict, tuple, set)) else x))

            X_df = sanitize(X_df)
            y_df = sanitize(y_df)

            if filename is None:
                filename = f"{self.symbol}_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                X_df.to_excel(writer, sheet_name='features')
                y_df.to_excel(writer, sheet_name='targets')

                stats = []
                for col in X_df.columns:
                    stats.append({
                        'type': 'feature', 'colonne': col,
                        'moyenne': X_df[col].mean(), 'ecart_type': X_df[col].std(),
                        'min': X_df[col].min(), 'max': X_df[col].max(),
                        'nb_nan': X_df[col].isna().sum(), 'pct_nan': X_df[col].isna().mean() * 100
                    })
                for col in y_df.columns:
                    stats.append({
                        'type': 'target', 'colonne': col,
                        'moyenne': y_df[col].mean(), 'ecart_type': y_df[col].std(),
                        'min': y_df[col].min(), 'max': y_df[col].max(),
                        'nb_nan': y_df[col].isna().sum(), 'pct_nan': y_df[col].isna().mean() * 100
                    })
                pd.DataFrame(stats).to_excel(writer, sheet_name='stats', index=False)

                info = {
                    'symbol': [self.symbol], 'period': [self.period],
                    'start_date': [common_idx.min()], 'end_date': [common_idx.max()],
                    'n_rows': [len(common_idx)], 'n_features': [X_df.shape[1]], 'n_targets': [y_df.shape[1]],
                    'feature_columns': [', '.join(X_df.columns[:5]) + '...'],
                    'target_columns': [', '.join(y_df.columns[:5]) + '...']
                }
                pd.DataFrame(info).to_excel(writer, sheet_name='info', index=False)

            logger.info(f"✅ Données exportées dans {filename}")
            return filename

        except Exception as e:
            logger.error(f"Erreur export Excel: {e}", exc_info=True)
            return None

    def train(self, epochs: int = 50, lookback_days: int = None, force_retrain: bool = False) -> bool:
        """
        Entraîne le modèle de prédiction.
        Si force_retrain=False, tente de charger un modèle existant compatible.
        """
        if not force_retrain and self.load_model_if_exists():
            logger.info("Modèle compatible existant chargé, entraînement ignoré.")
            return True

        try:
            tf.keras.backend.clear_session()
            gc.collect()

            if lookback_days is not None:
                self.lookback_days = lookback_days
                self.sequence_length = lookback_days

            model_dir = f"models_saved/{self.symbol}"
            checkpoints_dir = f"checkpoints/{self.symbol}"
            log_dir = f"logs/{self.symbol}"
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(checkpoints_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

            self._check_gpu()

            X_train, y_train, X_test, y_test = self.prepare_training_data(lookback_days=self.lookback_days)

            if X_train is None or X_train.shape[0] == 0:
                logger.error("Données insuffisantes pour l'entraînement")
                return False

            if X_train.shape[0] < self.batch_size * 2:
                new_batch_size = max(1, X_train.shape[0] // 4)
                logger.warning(f"Batch size ajusté de {self.batch_size} à {new_batch_size}")
                self.batch_size = new_batch_size

            n_outputs = y_train.shape[1]
            logger.info(f"Données d'entraînement: {X_train.shape[0]} échantillons, {n_outputs} cibles")
            logger.info(f"Données de test: {X_test.shape[0]} échantillons")

            # Architecture plus puissante
            self.model = ImprovedLSTMPredictorMultiOutput(
                lstm_units1=128,
                lstm_units2=64,
                lstm_units3=32,
                dense_units=64,
                dropout_rate=0.3,
                recurrent_dropout=0.2,
                l2_reg=0.001,
                n_outputs=n_outputs,
                use_attention=True,
                use_residual=True,
                attention_heads=4
            )

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                                                 epsilon=1e-07, clipnorm=1.0)
            self.model.compile(optimizer=optimizer, loss="huber", metrics=["mae", "mse"])

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=f"{checkpoints_dir}/best_model_{timestamp}.keras",
                    monitor='val_loss', save_best_only=True, mode='min', verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=self.patience_early_stopping,
                    min_delta=0.0001, restore_best_weights=True, verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.2, patience=8, min_lr=1e-7, verbose=1
                ),
                tf.keras.callbacks.TensorBoard(log_dir=f"{log_dir}/{timestamp}", histogram_freq=1)
            ]

            logger.info(f"Début de l'entraînement pour {epochs} epochs")
            start_time = datetime.now()
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs, batch_size=self.batch_size,
                callbacks=callbacks, shuffle=False, verbose=1
            )
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Temps d'entraînement: {training_time:.2f} secondes")

            # Validation sur test (prix réels après exponentiation)
            self.validate_on_test(X_test, y_test)

            # Sauvegarde finale
            model_path = f"{model_dir}/{self.symbol}_model_{timestamp}.keras"
            self.model.save(model_path)
            logger.info(f"Modèle sauvegardé: {model_path}")

            with open(f"{model_dir}/{self.symbol}_feature_scaler_{timestamp}.pkl", 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            with open(f"{model_dir}/{self.symbol}_target_scaler_{timestamp}.pkl", 'wb') as f:
                pickle.dump(self.target_scaler, f)

            # Métadonnées
            metadata = {
                'symbol': self.symbol, 'timestamp': timestamp,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns,
                'lookback_days': self.lookback_days,
                'use_log_target': True
            }
            with open(f"{model_dir}/{self.symbol}_metadata_{timestamp}.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            # Sauvegarde des métriques et graphique
            metrics = {
                'symbol': self.symbol, 'timestamp': timestamp,
                'training_time_seconds': training_time,
                'train_samples': len(X_train), 'test_samples': len(X_test),
                'lookback_days': self.lookback_days, 'batch_size': self.batch_size,
                'architecture': 'LSTM_128_64_32'
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
        """Évalue les performances réelles sur l'ensemble de test (prix réels après exponentiation)"""
        if self.model is None:
            logger.error("Modèle non entraîné")
            return

        y_pred_scaled = self.model.predict(X_test_seq, verbose=0)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        y_true = self.target_scaler.inverse_transform(y_test_seq)

        # Retour au prix réel (exp du log)
        for i, col in enumerate(self.target_columns):
            if 'Target_Close' in col:
                y_pred[:, i] = np.exp(y_pred[:, i])
                y_true[:, i] = np.exp(y_true[:, i])

        logger.info("=== Validation sur données de test (prix réels) ===")
        for i, col in enumerate(self.target_columns):
            if 'Target_Close' in col:
                mae = np.mean(np.abs(y_pred[:, i] - y_true[:, i]))
                mape = np.mean(np.abs((y_true[:, i] - y_pred[:, i]) / y_true[:, i])) * 100
                logger.info(f"{col}: MAE = ${mae:.2f}, MAPE = {mape:.1f}%")
                for j in range(min(3, len(y_pred))):
                    logger.info(f"  Exemple {j}: Prédit=${y_pred[j, i]:.2f}, Réel=${y_true[j, i]:.2f}")

    def train_on_arrays(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                        epochs: int = 50, batch_size: int = 32,
                        model_dir: Optional[str] = None, callbacks: Optional[List[Any]] = None) -> Any:
        """Entraîne le modèle à partir de tableaux numpy pré-préparés."""
        try:
            X_train, y_train = X_train.astype(np.float32), y_train.astype(np.float32)
            if X_val is not None:
                X_val = X_val.astype(np.float32)
            if y_val is not None:
                y_val = y_val.astype(np.float32)
            self._check_gpu()

            cb = callbacks or []
            cb += [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience_early_stopping,
                                                 restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=max(3, self.patience_early_stopping // 4),
                                                     factor=0.2)
            ]

            n_features, n_outputs = X_train.shape[2], y_train.shape[1]
            model = ImprovedLSTMPredictorMultiOutput(
                lstm_units1=128, lstm_units2=64, lstm_units3=32,
                dense_units=64, dropout_rate=0.3, recurrent_dropout=0.2,
                l2_reg=0.001, n_outputs=n_outputs
            )
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="huber", metrics=["mae", "mse"])

            history = model.fit(X_train, y_train,
                                validation_data=(X_val, y_val) if (X_val is not None and y_val is not None) else None,
                                epochs=epochs, batch_size=batch_size, callbacks=cb, shuffle=True, verbose=1)
            self.model = model
            self._save_model_and_scalers(model, model_dir)
            return history
        except Exception as e:
            logger.error(f"Erreur train_on_arrays: {e}", exc_info=True)
            raise

    def _save_model_and_scalers(self, model, model_dir):
        """Utilitaire de sauvegarde centralisé"""
        if model_dir is None:
            model_dir = f"models_saved/{self.symbol or 'batch'}"
        os.makedirs(model_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model.save(os.path.join(model_dir, f"model_{ts}.keras"))
        try:
            with open(os.path.join(model_dir, f"feature_scaler_{ts}.pkl"), "wb") as f:
                pickle.dump(self.feature_scaler, f)
            with open(os.path.join(model_dir, f"target_scaler_{ts}.pkl"), "wb") as f:
                pickle.dump(self.target_scaler, f)
            logger.info(f"Modèle et scalers sauvegardés dans {model_dir}")
        except Exception as e:
            logger.debug(f"Erreur sauvegarde scalers: {e}")

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

    def load_model_if_exists(self) -> bool:
        """
        Recherche et charge le modèle le plus récent pour ce symbole (spécifique ou batch).
        Vérifie la compatibilité des colonnes features avant d'accepter le modèle.
        Retourne True si un modèle compatible a été chargé.
        """

        try:
            import glob
            model_dirs = [f"models_saved/{self.symbol}", "models_saved/batch"]
            candidates = []
            for d in model_dirs:
                if os.path.isdir(d):
                    candidates.extend(glob.glob(os.path.join(d, "*.keras")))
            if not candidates:
                return False

            latest_model = max(candidates, key=os.path.getmtime)
            self.model = tf.keras.models.load_model(latest_model)
            logger.info(f"Modèle chargé depuis {latest_model}")

            scaler_dir = os.path.dirname(latest_model)

            # --- RECHERCHE DES SCALERS (avec priorité au préfixe symbole) ---
            # Feature scaler
            feat_patterns = [
                f"{self.symbol}_feature_scaler_*.pkl",
                "feature_scaler_*.pkl"
            ]
            feat_file = None
            for pattern in feat_patterns:
                matches = glob.glob(os.path.join(scaler_dir, pattern))
                if matches:
                    feat_file = max(matches, key=os.path.getmtime)
                    break
            if feat_file:
                with open(feat_file, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                logger.info(f"Feature scaler chargé depuis {feat_file}")
            else:
                logger.warning("Aucun feature scaler trouvé, le scaler restera non fitté")

            # Target scaler
            targ_patterns = [
                f"{self.symbol}_target_scaler_*.pkl",
                "target_scaler_*.pkl"
            ]
            targ_file = None
            for pattern in targ_patterns:
                matches = glob.glob(os.path.join(scaler_dir, pattern))
                if matches:
                    targ_file = max(matches, key=os.path.getmtime)
                    break
            if targ_file:
                with open(targ_file, 'rb') as f:
                    self.target_scaler = pickle.load(f)
                logger.info(f"Target scaler chargé depuis {targ_file}")
            else:
                logger.warning("Aucun target scaler trouvé")

            # Métadonnées (optionnel)
            meta_patterns = [
                f"{self.symbol}_metadata_*.json",
                "*metadata_*.json"
            ]
            meta_file = None
            for pattern in meta_patterns:
                matches = glob.glob(os.path.join(scaler_dir, pattern))
                if matches:
                    meta_file = max(matches, key=os.path.getmtime)
                    break
            if meta_file:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                self.feature_columns = meta.get('feature_columns', [])
                self.target_columns = meta.get('target_columns', [])
                self.lookback_days = meta.get('lookback_days', self.lookback_days)
                logger.info("Métadonnées chargées")

            # Vérification de compatibilité
            if self.features is not None and self.feature_columns:
                if not set(self.feature_columns).issubset(set(self.features.columns)):
                    logger.warning("Modèle incompatible: colonnes features manquantes")
                    self.model = None
                    return False

            logger.info("Modèle chargé et compatible avec les données actuelles.")
            return True

        except Exception as e:
            logger.error(f"Erreur load_model_if_exists: {e}", exc_info=True)
            return False

    def generate_predictions(self) -> Dict[str, Any]:
        """Génère les prédictions pour les horizons 1d,5d,10d,20d,30d,90d (prix réels)"""
        try:
            if self.model is None or self.features is None:
                logger.error("Modèle ou features non disponibles")
                return {}

            if not self.feature_columns:
                logger.error("Aucune colonne de features sauvegardée.")
                return {}

            missing = set(self.feature_columns) - set(self.features.columns)
            if missing:
                logger.error(f"Colonnes manquantes dans features: {missing}")
                return {}

            features_aligned = self.features[self.feature_columns]
            if len(features_aligned) < self.lookback_days:
                logger.error(f"Pas assez de données: besoin de {self.lookback_days} jours, disponible {len(features_aligned)}")
                return {}

            recent_data = features_aligned.iloc[-self.lookback_days:]
            scaled_data = self.feature_scaler.transform(recent_data)
            X_pred = scaled_data.reshape(1, self.lookback_days, -1)

            y_pred_scaled = self.model.predict(X_pred, verbose=0)
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled)

            # Exponentiation pour obtenir les prix réels
            pred_dict = {}
            for i, col in enumerate(self.target_columns):
                if 'Target_Close' in col:
                    pred_dict[col] = float(np.exp(y_pred[0, i]))
                else:
                    pred_dict[col] = float(y_pred[0, i])

            horizons = [1, 5, 10, 20, 30, 90]
            predictions = {}
            for days in horizons:
                col = f'Target_Close_{days}d'
                predictions[f'{days}d'] = pred_dict.get(col, None)

            logger.info(f"Prédictions extraites: {predictions}")
            return predictions

        except Exception as e:
            logger.error(f"Erreur generate_predictions: {e}")
            return {}

    def calculate_trading_score(self, predictions: Dict[str, Any]) -> float:
        """Calcule un score de trading basé sur les prédictions"""
        try:
            if not predictions or self.current_price <= 0:
                return 5.0
            returns = {}
            for horizon, pred in predictions.items():
                if pred is not None:
                    returns[horizon] = (pred - self.current_price) / self.current_price * 100

            weights = {"1d": 0.1, "5d": 0.15, "10d": 0.2, "30d": 0.3, "90d": 0.25}
            weighted_return = 0.0
            total_weight = 0.0
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
        """Génère une recommandation basée sur le score"""
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

    def analyze_model_stock(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Analyse complète de l'action.
        Si force_retrain=True, ignore les modèles existants et réentraîne.
        """
        try:
            logger.info(f"Analyse de {self.symbol}")

            # 1) Récupération des données
            if not self.fetch_data():
                return {"error": "Échec de la récupération des données", "symbol": self.symbol}

            # 2) Chargement ou entraînement
            if force_retrain:
                logger.info("Forçage du ré-entraînement...")
                if not self.train(epochs=50, force_retrain=True):
                    return {"error": "Échec de l'entraînement", "symbol": self.symbol}
            else:
                if not self.load_model_if_exists():
                    logger.info("Aucun modèle compatible trouvé, entraînement...")
                    if not self.train(epochs=50):
                        return {"error": "Échec de l'entraînement", "symbol": self.symbol}

            # 3) Prédictions et score
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
                "period": self.period,
                "model_loaded": (not force_retrain and self.model is not None)
            }
            return self.analysis_results

        except Exception as e:
            logger.error(f"Erreur analyze_model_stock: {e}", exc_info=True)
            return {"error": str(e), "symbol": self.symbol}

    def create_dashboard(self):
        """Crée un tableau de bord interactif (affichage console) - à adapter si nécessaire"""
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

    # ------------------------------------------------------------------
    # Méthodes pour l'entraînement par générateur (streaming) - inchangées
    # ------------------------------------------------------------------
    @staticmethod
    def _sequence_batch_generator_from_arrays(X: np.ndarray, y: np.ndarray,
                                               lookback_days: int, batch_size: int):
        """Yield batches of sliding windows from X,y without allocating the full tensor."""
        if X is None or y is None:
            return
        n = len(X)
        if n <= lookback_days:
            return
        buf_X = []
        buf_y = []
        for i in range(lookback_days, n):
            buf_X.append(X[i - lookback_days:i].astype(np.float32))
            buf_y.append(y[i].astype(np.float32))
            if len(buf_X) >= batch_size:
                yield np.stack(buf_X), np.stack(buf_y)
                buf_X = []
                buf_y = []
        if buf_X:
            yield np.stack(buf_X), np.stack(buf_y)

    @staticmethod
    def combined_data_generator(collected: Dict[str, Dict[str, pd.DataFrame]],
                                common_feature_cols: List[str], common_target_cols: List[str],
                                feature_scaler, target_scaler, lookback_days: int, batch_size: int):
        """
        Générateur infini pour model.fit().
        Parcourt les symboles et produit des batches (X, y) en boucle.
        """
        scaled_map = {}
        for sym, data in collected.items():
            X_df = data['features'][common_feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            y_df = data['targets'][common_target_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            if len(X_df) <= lookback_days or len(y_df) <= lookback_days:
                continue
            X_scaled = feature_scaler.transform(X_df.values).astype(np.float32)
            y_scaled = target_scaler.transform(y_df.values).astype(np.float32)
            scaled_map[sym] = (X_scaled, y_scaled)

        if not scaled_map:
            dummy_X = np.zeros((1, lookback_days, len(common_feature_cols)), dtype=np.float32)
            dummy_y = np.zeros((1, len(common_target_cols)), dtype=np.float32)
            while True:
                yield dummy_X, dummy_y

        while True:
            for sym, (X_s, y_s) in scaled_map.items():
                n_samples = len(X_s)
                for start_idx in range(lookback_days, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    batch_X = []
                    batch_y = []
                    for t in range(start_idx, end_idx):
                        batch_X.append(X_s[t - lookback_days:t])
                        batch_y.append(y_s[t])
                    if batch_X:
                        yield np.array(batch_X, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    def train_with_generator(self, data_gen, steps_per_epoch: int,
                             validation_gen=None, validation_steps=None,
                             epochs: int = 50, batch_size: int = 32,
                             model_dir: Optional[str] = None,
                             additional_callbacks: Optional[List[Any]] = None,
                             model_config: Optional[Dict[str, Any]] = None,
                             use_checkpoint: bool = True):
        """Entraîne le modèle à partir d'un générateur Python (streaming)."""
        try:
            self._check_gpu()
            n_features, n_outputs = self._infer_dims_from_generator(data_gen)
            if n_features is None or n_outputs is None:
                raise RuntimeError("Impossible de déterminer les dimensions.")

            if model_dir is None:
                model_dir = f"models_saved/{self.symbol or 'batch'}"
            os.makedirs(model_dir, exist_ok=True)

            best_model_path = self._find_latest_checkpoint(model_dir, pattern="best_model.keras")
            loaded_from_checkpoint = False
            if best_model_path and use_checkpoint:
                try:
                    self.model = tf.keras.models.load_model(best_model_path)
                    logger.info(f"Reprise depuis le meilleur modèle existant : {best_model_path}")
                    loaded_from_checkpoint = True
                    scaler_pattern = os.path.join(model_dir, "feature_scaler_*.pkl")
                    import glob
                    latest_scaler = max(glob.glob(scaler_pattern), key=os.path.getctime, default=None)
                    if latest_scaler:
                        with open(latest_scaler, "rb") as f:
                            self.feature_scaler = pickle.load(f)
                        with open(latest_scaler.replace("feature_scaler", "target_scaler"), "rb") as f:
                            self.target_scaler = pickle.load(f)
                        logger.info("Scalers rechargés depuis la dernière sauvegarde.")
                except Exception as e:
                    logger.warning(f"Impossible de charger le modèle existant : {e}")

            if not loaded_from_checkpoint:
                default_config = {
                    'lstm_units1': 128, 'lstm_units2': 64, 'lstm_units3': 32,
                    'dense_units': 64, 'dropout_rate': 0.3, 'recurrent_dropout': 0.2,
                    'l2_reg': 0.001, 'n_outputs': n_outputs
                }
                if model_config:
                    default_config.update(model_config)
                default_config['n_outputs'] = n_outputs
                model = ImprovedLSTMPredictorMultiOutput(**default_config)
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
                model.compile(optimizer=optimizer, loss="huber", metrics=["mae", "mse"])
                self.model = model

            cbs = additional_callbacks[:] if additional_callbacks else []
            checkpoint_path = None
            if use_checkpoint:
                checkpoint_path = os.path.join(model_dir, "best_model.keras")
                cbs.append(tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1
                ))
                logger.info(f"Le meilleur modèle sera sauvegardé dans {checkpoint_path}")

            cbs.extend([
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience_early_stopping,
                                                 restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                     patience=max(3, self.patience_early_stopping // 4), factor=0.2)
            ])

            train_gen = data_gen() if callable(data_gen) else data_gen
            val_gen = validation_gen() if callable(validation_gen) else validation_gen

            history = self.model.fit(train_gen, steps_per_epoch=steps_per_epoch,
                                     validation_data=val_gen, validation_steps=validation_steps,
                                     epochs=epochs, callbacks=cbs, verbose=1)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(model_dir, f"model_final_{ts}.keras")
            self.model.save(final_path)
            try:
                with open(os.path.join(model_dir, f"feature_scaler_{ts}.pkl"), "wb") as f:
                    pickle.dump(self.feature_scaler, f)
                with open(os.path.join(model_dir, f"target_scaler_{ts}.pkl"), "wb") as f:
                    pickle.dump(self.target_scaler, f)
            except Exception as e:
                logger.debug(f"Impossible de sauvegarder les scalers : {e}")
            logger.info(f"Entraînement terminé. Modèle final sauvegardé : {final_path}")
            return history

        except Exception as e:
            logger.error(f"Erreur dans train_with_generator : {e}", exc_info=True)
            if use_checkpoint and model_dir:
                best_model_path = self._find_latest_checkpoint(model_dir, pattern="best_model.keras")
                if best_model_path and os.path.exists(best_model_path):
                    try:
                        self.model = tf.keras.models.load_model(best_model_path)
                        logger.info(f"Crash détecté – meilleur modèle restauré depuis {best_model_path}")
                    except Exception as load_err:
                        logger.error(f"Impossible de restaurer le meilleur modèle : {load_err}")
            raise

    def _find_latest_checkpoint(self, model_dir: str, pattern: str = "best_model_*.keras") -> Optional[str]:
        import glob
        files = glob.glob(os.path.join(model_dir, pattern))
        return max(files, key=os.path.getmtime) if files else None

    def _infer_dims_from_generator(self, data_gen):
        try:
            gen = data_gen() if callable(data_gen) else data_gen
            batch_X, batch_y = next(iter(gen))
            if len(batch_X.shape) != 3:
                raise ValueError(f"Le batch_X doit être 3D, shape={batch_X.shape}")
            n_features = batch_X.shape[2]
            n_outputs = batch_y.shape[1]
            logger.info(f"Dimensions inférées: n_features={n_features}, n_outputs={n_outputs}")
            return n_features, n_outputs
        except Exception as e:
            logger.error(f"Erreur lors de l'inférence des dimensions: {e}", exc_info=True)
            return None, None


# ----------------------------------------------------------------------
# Point d'entrée principal (CLI)
# ----------------------------------------------------------------------
def main():
    """Fonction principale d'exécution dans stockmodeltrain"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("🚀 PRÉDICTEUR D'ACTIONS AVEC IA (version optimisée)")
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

    force_retrain = input("\nForcer un ré-entraînement complet ? (o/N): ").strip().lower() == 'o'

    logger.info(f"\n{'='*60}")
    logger.info(f"Lancement de l'analyse pour {symbol} ({period})...")
    logger.info(f"{'='*60}")

    start_time = datetime.now()
    try:
        predictor = StockModelTrain(symbol, period)
        results = predictor.analyze_model_stock(force_retrain=force_retrain)
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


if __name__ == "__main__":
    main()