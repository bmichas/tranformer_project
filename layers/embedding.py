import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding
import keras


def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)        
    angle_rads = positions * angle_rates      
    sin = np.sin(angle_rads)
    cos = np.cos(angle_rads)
    pos_encoding = np.concatenate([sin, cos],axis=-1) 
    return tf.cast(pos_encoding, dtype=tf.float32)


@keras.saving.register_keras_serializable(package="PositionalEmbedding")
class PositionalEmbedding(Layer):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = Embedding(vocab_size, emb_dim, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=2048, depth=emb_dim)


    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)


    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.emb_dim, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x