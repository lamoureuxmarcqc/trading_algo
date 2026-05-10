# -*- coding: utf-8 -*-
"""
Module fusionné et amélioré pour la sélection et la recherche d'architectures de modèles ML/DL.
Supporte : classification, régression simple, régression multi-output.
Contient :
 - Définitions de modèles Keras personnalisés (LSTM avancé, hybrid CNN-LSTM...)
 - Utilitaires (scaling, callbacks, métriques)
 - `ModelSelector` : pipeline complet de comparaison de modèles scikit-learn + réseaux
 - `ModelFinder` : recherche simple d'architecture pour modèles LSTM avancés
 - Fonction utilitaire `find_best_model` pour usage direct
Les messages de logs sont en français pour cohérence avec le projet.
"""
from __future__ import annotations

import os
import warnings
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# Environment / TF settings to reduce verbose logs and stabilize behavior
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import pickle

# Scikit-learn
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score

# XGBoost (optional dependency)
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = XGBRegressor = None

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Flatten, Conv1D, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, TerminateOnNaN

# Try to import project utilitaire if exists
try:
    from . import utilitaire as util
except Exception:
    util = None

# ----------------------------
# Modèles Keras personnalisés
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class ImprovedLSTMPredictorMultiOutput(keras.Model):
    """LSTM avancé avec attention pour tâches (multi-)régression ou classification."""
    def __init__(
        self,
        lstm_units1: int = 64,
        lstm_units2: int = 32,
        dense_units: int = 32,
        dropout_rate: float = 0.1,
        recurrent_dropout: float = 0.1,
        l2_reg: float = 1e-4,
        n_outputs: int = 1,
        use_bidirectional: bool = False,
        output_activation: str = "linear",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lstm_units1 = lstm_units1
        self.lstm_units2 = lstm_units2
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.l2_reg = l2_reg
        self.n_outputs = n_outputs
        self.use_bidirectional = use_bidirectional
        self.output_activation = output_activation

        lstm_layer1 = LSTM(
            lstm_units1,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )
        lstm_layer2 = LSTM(
            lstm_units2,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )

        if use_bidirectional:
            self.lstm1 = Bidirectional(lstm_layer1)
            self.lstm2 = Bidirectional(lstm_layer2)
        else:
            self.lstm1 = lstm_layer1
            self.lstm2 = lstm_layer2

        self.attention = layers.Attention()
        self.dense1 = Dense(dense_units, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg))
        self.batch_norm = BatchNormalization()
        self.dropout = Dropout(0.3)
        self.dense2 = Dense(max(8, dense_units // 2), activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg))
        self.output_dense = Dense(n_outputs, activation=output_activation)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False, mask=None):
        x = self.lstm1(inputs, training=training, mask=mask)
        x = self.lstm2(x, training=training)
        attention_output = self.attention([x, x])
        x = tf.reduce_mean(attention_output, axis=1)
        x = self.dense1(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.layer_norm(x)
        return self.output_dense(x)

    def get_config(self):
        cfg = super().get_config() if hasattr(super(), "get_config") else {}
        cfg.update({
            "lstm_units1": self.lstm_units1,
            "lstm_units2": self.lstm_units2,
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate,
            "recurrent_dropout": self.recurrent_dropout,
            "l2_reg": self.l2_reg,
            "n_outputs": self.n_outputs,
            "use_bidirectional": self.use_bidirectional,
            "output_activation": self.output_activation,
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ImprovedLSTMPredictor(ImprovedLSTMPredictorMultiOutput):
    """Alias / version compatible quand n_outputs == 1"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@tf.keras.utils.register_keras_serializable()
class HybridCNNLSTMModel(keras.Model):
    def __init__(self, conv_filters=(32, 64), lstm_units=64, dense_units=32, dropout_rate=0.2, n_outputs=1, output_activation="linear", **kwargs):
        super().__init__(**kwargs)
        self.conv_layers = []
        for filters in conv_filters:
            self.conv_layers.append(Conv1D(filters, 3, padding="same", activation="relu"))
            self.conv_layers.append(BatchNormalization())
            self.conv_layers.append(MaxPooling1D(1))
        self.lstm = LSTM(lstm_units, return_sequences=False, dropout=dropout_rate)
        self.dense1 = Dense(dense_units, activation="relu")
        self.dropout = Dropout(dropout_rate)
        self.dense2 = Dense(max(8, dense_units // 2), activation="relu")
        self.output_layer = Dense(n_outputs, activation=output_activation)

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.conv_layers:
            if isinstance(layer, (Conv1D, BatchNormalization)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        x = self.lstm(x, training=training)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)


# ----------------------------
# Utilitaires (scaling, callbacks, métriques)
# ----------------------------
def create_advanced_callbacks(model_name: str, monitor: str = "val_loss", patience: int = 15, min_lr: float = 1e-7, save_best_only: bool = True, log_dir: str = "logs"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, verbose=1, mode="auto"),
        ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=max(5, patience // 3), min_lr=min_lr, verbose=1, mode="auto"),
        ModelCheckpoint(filepath=f"best_{model_name}_{timestamp}.keras", monitor=monitor, save_best_only=save_best_only, verbose=1, save_format="tf"),
        TerminateOnNaN(),
        TensorBoard(log_dir=f"{log_dir}/{model_name}_{timestamp}", histogram_freq=1, write_graph=True, write_images=False),
    ]
    return callbacks


def scale_features(features: np.ndarray, scaler: Optional[StandardScaler] = None, fit: bool = False) -> Tuple[np.ndarray, StandardScaler]:
    original_shape = features.shape
    if features.ndim > 2:
        features_2d = features.reshape(-1, features.shape[-1])
    else:
        features_2d = features
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        scaled_features_2d = scaler.fit_transform(features_2d)
    else:
        scaled_features_2d = scaler.transform(features_2d)
    scaled_features = scaled_features_2d.reshape(original_shape)
    return scaled_features, scaler


def prepare_classification_data(y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    metadata: Dict[str, Any] = {}
    y = y.flatten()
    unique_classes = np.unique(y)
    metadata["num_classes"] = len(unique_classes)
    metadata["class_names"] = unique_classes.tolist()
    class_weights = class_weight.compute_class_weight("balanced", classes=unique_classes, y=y)
    metadata["class_weights"] = dict(enumerate(class_weights))
    return y, metadata


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, problem_type: str):
    metrics = {}
    if problem_type == "classification":
        y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
        y_true_classes = y_true
        metrics["accuracy"] = accuracy_score(y_true_classes, y_pred_classes)
        metrics["f1_weighted"] = f1_score(y_true_classes, y_pred_classes, average="weighted")
        metrics["f1_macro"] = f1_score(y_true_classes, y_pred_classes, average="macro")
        if len(np.unique(y_true_classes)) == 2:
            metrics["precision"] = precision_score(y_true_classes, y_pred_classes)
            metrics["recall"] = recall_score(y_true_classes, y_pred_classes)
    elif problem_type == "regression":
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        metrics["mape"] = mape
    elif problem_type == "multioutput_regression":
        metrics["mse"] = mean_squared_error(y_true, y_pred, multioutput="uniform_average")
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
        metrics["r2"] = r2_score(y_true, y_pred, multioutput="uniform_average")
        r2_scores = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
        metrics["r2_per_output"] = r2_scores
        metrics["r2_mean"] = np.mean(r2_scores)
    return metrics


def create_model_factory(model_type: str, input_shape: Tuple[int, ...], problem_type: str = "classification", n_outputs: int = 1, **kwargs):
    if problem_type == "classification":
        num_classes = kwargs.get("num_classes", n_outputs)
        output_units = num_classes
        output_activation = "softmax" if num_classes > 1 else "sigmoid"
    else:
        output_units = n_outputs
        output_activation = "linear"

    if model_type == "dense":
        model = Sequential([Input(shape=input_shape), Flatten(),
                            Dense(128, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                            Dropout(0.3), Dense(64, activation="relu"), Dropout(0.2), Dense(32, activation="relu"),
                            Dense(output_units, activation=output_activation)])
    elif model_type == "lstm":
        model = Sequential([Input(shape=input_shape),
                            LSTM(64, return_sequences=True, dropout=0.2),
                            LSTM(32, dropout=0.2),
                            Dense(32, activation="relu"), Dropout(0.2),
                            Dense(output_units, activation=output_activation)])
    elif model_type == "cnn":
        model = Sequential([Input(shape=input_shape),
                            Conv1D(32, 3, activation="relu", padding="same"),
                            BatchNormalization(), MaxPooling1D(1),
                            Conv1D(64, 3, activation="relu", padding="same"),
                            BatchNormalization(), GlobalAveragePooling1D(),
                            Dense(64, activation="relu"), Dropout(0.3),
                            Dense(output_units, activation=output_activation)])
    elif model_type == "hybrid":
        model = HybridCNNLSTMModel(conv_filters=[32, 64], lstm_units=64, dense_units=32, dropout_rate=0.2, n_outputs=output_units, output_activation=output_activation)
    elif model_type == "improved_lstm":
        model = ImprovedLSTMPredictorMultiOutput(n_outputs=output_units, output_activation=output_activation, **{k: v for k, v in kwargs.items() if k in ["lstm_units1", "lstm_units2", "dense_units", "dropout_rate", "recurrent_dropout", "l2_reg", "use_bidirectional"]})
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
    return model


def get_optimizer(optimizer_name: str = "adam", learning_rate: float = 0.001):
    if optimizer_name.lower() == "adam":
        return Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, clipnorm=1.0)
    elif optimizer_name.lower() == "rmsprop":
        return RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-7)
    else:
        return Adam(learning_rate=learning_rate)


# ----------------------------
# ModelSelector : pipeline complet
# ----------------------------
class ModelSelector:
    def __init__(self, problem_type: str = "classification", random_state: int = 42, n_jobs: int = -1, verbose: int = 1):
        self.problem_type = problem_type
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.feature_scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None

        self.metadata: Dict[str, Any] = {}
        self.best_model: Any = None
        self.best_model_name: str = ""
        self.best_score: float = -np.inf
        self.all_scores: Dict[str, Any] = {}

        self._setup_models()

    def _setup_models(self):
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        if self.problem_type == "classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            }
            if XGBClassifier is not None:
                models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=self.random_state, eval_metric="logloss")
            models["MLP"] = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=self.random_state)
            self.models = models
        elif self.problem_type == "regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(random_state=self.random_state),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            }
            if XGBRegressor is not None:
                models["XGBoost Regressor"] = XGBRegressor(n_estimators=100, random_state=self.random_state)
            self.models = models
        elif self.problem_type == "multioutput_regression":
            models = {
                "MultiOutput Linear Regression": MultiOutputRegressor(LinearRegression()),
                "MultiOutput Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=self.random_state)),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            }
            if XGBRegressor is not None:
                models["XGBoost Regressor"] = XGBRegressor(n_estimators=100, random_state=self.random_state)
            self.models = models
        else:
            raise ValueError(f"Unknown problem_type: {self.problem_type}")

    def preprocess_data(self, X_train, y_train, X_test=None, y_test=None, fit_scalers: bool = True):
        X_train_scaled, self.feature_scaler = scale_features(X_train, self.feature_scaler, fit=fit_scalers)
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled, _ = scale_features(X_test, self.feature_scaler, fit=False)

        if self.problem_type == "classification":
            y_train_processed, metadata = prepare_classification_data(y_train)
            self.metadata.update(metadata)
            y_test_processed = None
            if y_test is not None:
                y_test_processed, _ = prepare_classification_data(y_test)
        else:
            self.target_scaler = StandardScaler()
            if y_train.ndim == 1:
                y_train_2d = y_train.reshape(-1, 1)
            else:
                y_train_2d = y_train
            if fit_scalers:
                y_train_scaled = self.target_scaler.fit_transform(y_train_2d)
            else:
                y_train_scaled = self.target_scaler.transform(y_train_2d)
            y_train_processed = y_train_scaled.squeeze() if y_train_scaled.shape[1] == 1 else y_train_scaled
            y_test_processed = None
            if y_test is not None:
                y_test_2d = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test
                y_test_scaled = self.target_scaler.transform(y_test_2d)
                y_test_processed = y_test_scaled.squeeze() if y_test_scaled.shape[1] == 1 else y_test_scaled

        return X_train_scaled, y_train_processed, X_test_scaled, y_test_processed

    def _prepare_data_formats(self, X_train, X_test):
        data_formats = {}
        if X_train.ndim == 3:
            data_formats["2d"] = (X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1) if X_test is not None else None)
            data_formats["3d"] = (X_train, X_test)
        else:
            data_formats["2d"] = (X_train, X_test)
            data_formats["3d"] = (X_train.reshape(X_train.shape[0], 1, X_train.shape[1]), X_test.reshape(X_test.shape[0], 1, X_test.shape[1]) if X_test is not None else None)
        return data_formats

    def _select_data_format(self, model_name, data_formats):
        if any(k in model_name.lower() for k in ["lstm", "cnn", "neural", "improved", "hybrid"]):
            return data_formats.get("3d", data_formats["2d"])
        return data_formats["2d"]

    def _get_scoring_metric(self):
        if self.problem_type == "classification":
            return "accuracy"
        return "r2"

    def _inverse_transform_predictions(self, y_pred):
        if self.target_scaler is not None:
            if y_pred.ndim == 1:
                y_pred_2d = y_pred.reshape(-1, 1)
                return self.target_scaler.inverse_transform(y_pred_2d).flatten()
            else:
                return self.target_scaler.inverse_transform(y_pred)
        return y_pred

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, cv_folds: int = 5, neural_networks: bool = True):
        logging.info(f"Selection de modeles pour: {self.problem_type}")
        X_train_p, y_train_p, X_test_p, y_test_p = self.preprocess_data(X_train, y_train, X_test, y_test, fit_scalers=True)
        data_formats = self._prepare_data_formats(X_train_p, X_test_p)

        for name, model in self.models.items():
            try:
                logging.info(f"[Entrainement] {name}")
                train_data, test_data = self._select_data_format(name, data_formats)
                # cross_val expects 2d arrays
                cv_scores = cross_val_score(model, train_data, y_train_p, cv=cv_folds, n_jobs=self.n_jobs, scoring=self._get_scoring_metric())
                model.fit(train_data, y_train_p)
                y_pred = model.predict(test_data)
                if self.problem_type in ["regression", "multioutput_regression"]:
                    y_pred = self._inverse_transform_predictions(y_pred)
                    y_true = self._inverse_transform_predictions(y_test_p)
                else:
                    y_true = y_test
                metrics = calculate_comprehensive_metrics(y_true, y_pred, self.problem_type)
                main_score = metrics["accuracy"] if self.problem_type == "classification" else metrics.get("r2", np.nan)
                self.all_scores[name] = {"model": model, "cv_mean": float(np.mean(cv_scores)), "cv_std": float(np.std(cv_scores)), "metrics": metrics, "main_score": main_score}
                logging.info(f" CV Score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
                logging.info(f" Test Score: {main_score:.4f}")
                if main_score > self.best_score:
                    self.best_score = main_score
                    self.best_model = model
                    self.best_model_name = name
            except Exception as e:
                logging.error(f" Erreur avec {name}: {str(e)}")
                continue

        if neural_networks:
            self._train_neural_networks(data_formats, y_train_p, y_test_p, cv_folds)

        self._print_results()
        return {"best_model": self.best_model, "best_model_name": self.best_model_name, "best_score": self.best_score, "all_scores": self.all_scores, "feature_scaler": self.feature_scaler, "target_scaler": self.target_scaler, "metadata": self.metadata}

    def _train_neural_networks(self, data_formats, y_train, y_test, cv_folds):
        neural_models = [("Dense Neural Network", "dense"), ("LSTM Neural Network", "lstm"), ("CNN Neural Network", "cnn"), ("Hybrid CNN-LSTM", "hybrid"), ("Improved LSTM", "improved_lstm")]
        for name, model_type in neural_models:
            try:
                logging.info(f"[Entrainement NN] {name}")
                train_data, test_data = self._select_data_format(name, data_formats)
                kwargs = {}
                if self.problem_type == "classification":
                    kwargs["num_classes"] = self.metadata.get("num_classes", 2)
                    y_train_nn = to_categorical(y_train, num_classes=kwargs["num_classes"])
                    y_test_nn = to_categorical(y_test, num_classes=kwargs["num_classes"])
                    output_activation = "softmax" if kwargs["num_classes"] > 1 else "sigmoid"
                else:
                    y_train_nn = y_train
                    y_test_nn = y_test
                    output_activation = "linear"

                model = create_model_factory(model_type=model_type, input_shape=train_data.shape[1:], problem_type=self.problem_type, n_outputs=y_train_nn.shape[1] if (y_train_nn.ndim > 1) else 1, output_activation=output_activation, **kwargs)
                optimizer = get_optimizer("adam", 0.001)
                if self.problem_type == "classification":
                    loss = "categorical_crossentropy"
                    metrics = ["accuracy"]
                else:
                    loss = "mse"
                    metrics = ["mae", "mse"]
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
                callbacks = create_advanced_callbacks(f"neural_{model_type}")
                history = model.fit(train_data, y_train_nn, validation_data=(test_data, y_test_nn), epochs=50, batch_size=32, callbacks=callbacks, verbose=1, class_weight=self.metadata.get("class_weights", None))
                y_pred = model.predict(test_data, verbose=0)
                if self.problem_type in ["regression", "multioutput_regression"]:
                    y_pred_transformed = self._inverse_transform_predictions(y_pred)
                    y_true_transformed = self._inverse_transform_predictions(y_test)
                else:
                    y_pred_transformed = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
                    y_true_transformed = y_test
                metrics = calculate_comprehensive_metrics(y_true_transformed, y_pred_transformed, self.problem_type)
                main_score = metrics["accuracy"] if self.problem_type == "classification" else metrics.get("r2", np.nan)
                self.all_scores[name] = {"model": model, "history": history.history, "metrics": metrics, "main_score": main_score}
                logging.info(f" Validation Score: {main_score:.4f}")
                if main_score > self.best_score:
                    self.best_score = main_score
                    self.best_model = model
                    self.best_model_name = name
            except Exception as e:
                logging.error(f" Erreur NN avec {name}: {str(e)}")
                continue

    def _print_results(self):
        logging.info("\n============================================================\nRÉSULTATS DE LA SÉLECTION DE MODÈLES\n============================================================")
        sorted_scores = sorted(self.all_scores.items(), key=lambda x: x[1]["main_score"], reverse=True)
        logging.info("\nClassement des modèles:\n----------------------------------------")
        for rank, (name, results) in enumerate(sorted_scores, 1):
            score = results["main_score"]
            cv_info = ""
            if "cv_mean" in results:
                cv_info = f" | CV: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})"
            logging.info(f" {rank}. {name} : {score:.4f}{cv_info}")
        logging.info(f"\n============================================================\nMEILLEUR MODÈLE: {self.best_model_name}\nSCORE: {self.best_score:.4f}\n============================================================")

    def save_best_model(self, path: str = "best_model"):
        import joblib
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        # Prefer utilitaire save if available
        try:
            if util and hasattr(util, "save_model"):
                util.save_model(self.best_model, self.best_model_name)
        except Exception:
            logging.debug("util.save_model failed or not available; falling back to local saving.")

        if hasattr(self.best_model, "save"):
            try:
                self.best_model.save(f"{path}", save_format="tf")
                logging.info(f"✓ Modèle Keras sauvegardé (format SavedModel): {path}")
            except Exception:
                self.best_model.save_weights(f"{path}_weights.weights.h5")
                logging.info(f"✓ Poids du modèle Keras sauvegardés: {path}_weights.weights.h5")
                logging.warning("Note: L'architecture du modèle devra être recréée lors du chargement")
        else:
            joblib.dump(self.best_model, f"{path}.joblib")
            logging.info(f"✓ Modèle scikit-learn sauvegardé: {path}.joblib")

        if self.feature_scaler:
            joblib.dump(self.feature_scaler, f"{path}_feature_scaler.joblib")
        if self.target_scaler:
            joblib.dump(self.target_scaler, f"{path}_target_scaler.joblib")

        metadata = {"best_model_name": self.best_model_name, "best_score": self.best_score, "problem_type": self.problem_type, "metadata": self.metadata, "model_type": "keras" if hasattr(self.best_model, "save") else "sklearn"}
        joblib.dump(metadata, f"{path}_metadata.joblib")
        logging.info(f"✓ Métadonnées sauvegardées: {path}_metadata.joblib")

    def load_best_model(self, path: str = "best_model"):
        import joblib
        import tensorflow as tf
        metadata = joblib.load(f"{path}_metadata.joblib")
        if metadata["model_type"] == "keras":
            if os.path.exists(f"{path}"):
                model = tf.keras.models.load_model(f"{path}")
                logging.info(f"✓ Modèle Keras chargé: {path}")
            else:
                raise NotImplementedError("Chargement d'un modèle Keras par poids seul non implémenté automatiquement. Recréez l'architecture et chargez les poids.")
        else:
            model = joblib.load(f"{path}.joblib")
            logging.info(f"✓ Modèle scikit-learn chargé: {path}.joblib")
        feature_scaler = joblib.load(f"{path}_feature_scaler.joblib") if os.path.exists(f"{path}_feature_scaler.joblib") else None
        target_scaler = joblib.load(f"{path}_target_scaler.joblib") if os.path.exists(f"{path}_target_scaler.joblib") else None
        return model, feature_scaler, target_scaler, metadata


# ----------------------------
# ModelFinder : recherche d'architecture simple pour le LSTM amélioré
# ----------------------------
class ModelFinder:
    """Recherche d'architectures pour `ImprovedLSTMPredictorMultiOutput`."""
    def __init__(self):
        self.best_model = None
        self.best_score = float("inf")
        self.best_architecture = None

    def find_best_architecture(self, X_train, y_train, X_val, y_val, architectures: Optional[list] = None, epochs: int = 10, batch_size: int = 32, verbose: int = 0):
        if architectures is None:
            architectures = [
                {"lstm_units1": 32, "lstm_units2": 16, "dense_units": 16},
                {"lstm_units1": 64, "lstm_units2": 32, "dense_units": 32},
                {"lstm_units1": 128, "lstm_units2": 64, "dense_units": 64},
                {"lstm_units1": 256, "lstm_units2": 128, "dense_units": 64},
            ]
        for arch in architectures:
            model = ImprovedLSTMPredictorMultiOutput(
                lstm_units1=arch["lstm_units1"],
                lstm_units2=arch["lstm_units2"],
                dense_units=arch["dense_units"],
                n_outputs=y_train.shape[1] if y_train.ndim > 1 else 1,
            )
            model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=verbose)
            val_loss = float(history.history.get("val_loss", [np.inf])[-1])
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_model = model
                self.best_architecture = arch
        return self.best_model, self.best_architecture


# ----------------------------
# Fonction utilitaire pour usage rapide
# ----------------------------
def find_best_model(X_train, y_train, X_test, y_test, problem_type: str = "classification", **kwargs):
    """API rapide : détecte type si 'auto' et lance ModelSelector puis sauvegarde le meilleur modèle."""
    if problem_type == "auto":
        if y_train.ndim > 1 and y_train.shape[1] > 1:
            problem_type = "multioutput_regression"
        elif len(np.unique(y_train)) < 10:
            problem_type = "classification"
        else:
            problem_type = "regression"
    logging.info(f"Type de problème détecté: {problem_type}")
    selector = ModelSelector(problem_type=problem_type, **kwargs)
    results = selector.train_and_evaluate(X_train, y_train, X_test, y_test)
    try:
        selector.save_best_model()
    except Exception as e:
        logging.warning(f"Impossible de sauvegarder automatiquement le meilleur modèle: {e}")
    return results


# ----------------------------
# Test rapide si utilisé comme script
# ----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # petit test multi-output
    n_samples = 500
    X = np.random.randn(n_samples, 5, 4)
    y = np.zeros((n_samples, 3))
    for i in range(3):
        y[:, i] = X[:, :, 0].sum(axis=1) + np.random.randn(n_samples) * 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = find_best_model(X_train, y_train, X_test, y_test, problem_type="multioutput_regression", n_jobs=1, verbose=1)
    logging.info(f"Meilleur modèle: {results['best_model_name']} - score: {results['best_score']}")