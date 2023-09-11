import keras
import tensorflow as tf

from layers.decoder import Decoder
from layers.encoder import Encoder


@keras.saving.register_keras_serializable(package="Transformer")
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, emb_dim, num_heads, feed_forward,
                input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, emb_dim=emb_dim,
                            num_heads=num_heads, feed_forward=feed_forward,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, emb_dim=emb_dim,
                            num_heads=num_heads, feed_forward=feed_forward,
                            vocab_size=target_vocab_size,
                            dropout_rate=dropout_rate)

        self.dense = tf.keras.layers.Dense(feed_forward)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation = 'softmax')

    def call(self, inputs):
        context, x  = inputs
        context = self.encoder(context)
        x = self.decoder(x, context)
        x = self.dense(x)
        logits = self.final_layer(x)  
        return logits