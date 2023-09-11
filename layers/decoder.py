import keras
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout

from layers.attention import CausalSelfAttention, CrossAttention
from layers.embedding import PositionalEmbedding
from layers.feedforward import FeedForward


@keras.saving.register_keras_serializable(package="DecoderLayer")
class DecoderLayer(Layer):
    def __init__(self,
                *,
                emb_dim,
                num_heads,
                feed_forward,
                dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=emb_dim,
            dropout=dropout_rate)
        
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=emb_dim,
            dropout=dropout_rate)

        self.ffn = FeedForward(emb_dim, feed_forward)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)
        return x

@keras.saving.register_keras_serializable(package="Decoder")
class Decoder(Layer):
    def __init__(self, *, num_layers, emb_dim, num_heads, feed_forward, vocab_size,
                dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.emb_dim = emb_dim
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            emb_dim=emb_dim)
        
        self.dropout = Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                emb_dim=emb_dim, 
                num_heads=num_heads,
                feed_forward=feed_forward, dropout_rate=dropout_rate) 
                for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x  = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x