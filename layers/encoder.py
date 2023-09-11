import keras
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout

from layers.attention import GlobalSelfAttention
from layers.embedding import PositionalEmbedding
from layers.feedforward import FeedForward


@keras.saving.register_keras_serializable(package="EncoderLayer")
class EncoderLayer(Layer):
    def __init__(self,*, emb_dim, num_heads, feed_forward, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=emb_dim,
            dropout=dropout_rate)

        self.ffn = FeedForward(emb_dim, feed_forward)


    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


@keras.saving.register_keras_serializable(package="Encoder")
class Encoder(Layer):
    def __init__(self, *, num_layers, emb_dim, num_heads,
                feed_forward, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, emb_dim=emb_dim)

        self.enc_layers = [
            EncoderLayer(
                emb_dim=emb_dim,
                num_heads=num_heads,
                feed_forward=feed_forward,
                dropout_rate=dropout_rate) 
                for _ in range(num_layers)]
        
        self.dropout = Dropout(dropout_rate)


    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x