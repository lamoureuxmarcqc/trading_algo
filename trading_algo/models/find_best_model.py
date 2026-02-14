"""
Module pour trouver le meilleur modèle et définir ImprovedLSTMPredictorMultiOutput
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
tf.get_logger().setLevel('ERROR')

class ImprovedLSTMPredictorMultiOutput(keras.Model):
    """
    Modèle LSTM amélioré pour prédictions multi-sorties
    """
    def __init__(self, lstm_units1=64, lstm_units2=32, dense_units=32, 
                 dropout_rate=0.5, recurrent_dropout=0.2, n_outputs=6):
        super(ImprovedLSTMPredictorMultiOutput, self).__init__()
        
        self.lstm1 = layers.LSTM(
            units=lstm_units1,
            return_sequences=True,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=keras.regularizers.l2(0.01)
        )
        self.batch_norm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.lstm2 = layers.LSTM(
            units=lstm_units2,
            return_sequences=False,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=keras.regularizers.l2(0.01)
        )
        self.batch_norm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)
        
        self.dense1 = layers.Dense(dense_units, activation='relu', 
                                   kernel_regularizer=keras.regularizers.l2(0.01))
        self.batch_norm3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(dropout_rate)
        
        # Couche de sortie avec n_outputs cibles
        self.output_layer = layers.Dense(n_outputs, activation='linear')
        
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.lstm2(x)
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        
        x = self.dense1(x)
        x = self.batch_norm3(x, training=training)
        x = self.dropout3(x, training=training)
        
        return self.output_layer(x)


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