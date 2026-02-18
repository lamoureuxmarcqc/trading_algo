"""
Module pour trouver le meilleur modèle (dépend de base_model)
"""

import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel('ERROR')

from trading_algo.models.base_model import ImprovedLSTMPredictorMultiOutput

class ModelFinder:
    """
    Trouve le meilleur modèle basé sur les données
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = float('inf')
    
    def find_best_architecture(self, X_train, y_train, X_val, y_val):
        """Trouve la meilleure architecture de modèle"""
        architectures = [
            {'lstm_units1': 32, 'lstm_units2': 16, 'dense_units': 16},
            {'lstm_units1': 64, 'lstm_units2': 32, 'dense_units': 32},
            {'lstm_units1': 128, 'lstm_units2': 64, 'dense_units': 64},
            {'lstm_units1': 256, 'lstm_units2': 128, 'dense_units': 64}
        ]
        
        for arch in architectures:
            model = ImprovedLSTMPredictorMultiOutput(
                lstm_units1=arch['lstm_units1'],
                lstm_units2=arch['lstm_units2'],
                dense_units=arch['dense_units'],
                n_outputs=y_train.shape[1]
            )
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=32,
                verbose=0
            )
            
            val_loss = history.history['val_loss'][-1]
            
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_model = model
                self.best_architecture = arch
        
        return self.best_model, self.best_architecture