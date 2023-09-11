import keras
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Layer


@keras.saving.register_keras_serializable(package="BaseAttention")    
class BaseAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = MultiHeadAttention(**kwargs)
        self.layernorm = LayerNormalization()
        self.add = Add()


@keras.saving.register_keras_serializable(package="CrossAttention")
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)
    
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    

@keras.saving.register_keras_serializable(package="GlobalSelfAttention")
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    

@keras.saving.register_keras_serializable(package="CausalSelfAttention")
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x