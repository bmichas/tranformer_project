import keras
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Layer, Dropout, Dense, Add, LayerNormalization


@keras.saving.register_keras_serializable(package="FeedForward")
class FeedForward(Layer):
    def __init__(self, emb_dim, feed_forward, dropout_rate=0.1):
        super().__init__()

        self.seq = Sequential([
            Dense(feed_forward, activation='relu'),
            Dense(emb_dim),
            Dropout(dropout_rate)
            ])
        
        self.add = Add()
        self.layer_norm = LayerNormalization()


    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x