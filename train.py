from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf

from helpers.functions import unpickle
from transformer import Transformer



TOKENIZER = unpickle('data/tokenizer.pkl')
SEQUENCES = unpickle('data/sequences.pkl')
TARGETS = unpickle('data/targets.pkl')

class PredictionCallback(tf.keras.callbacks.Callback):    
    def _preprocess_input_text(self, text):
        text = text.split()
        sequence = text
        vector_sequences = []
        vector_sequence = []
        for word in sequence:
            vector_sequence.append(word)

        vector_sequences.append(vector_sequence)
        vector_sequences = TOKENIZER.texts_to_sequences(vector_sequences)
        return tf.convert_to_tensor(vector_sequences)
    

    def on_epoch_end(self, epoch, logs={}):
        """SAVING MODEL IF YOU WANT"""
        self.model.save('models/Transformer_'+ str('final')+ str(epoch)+'.keras')
        self.model.save_weights('models/Weights_Transformer_'+ str('final')+ str(epoch)+'.keras')
        num_words_to_generate = 200
        generated_text = 'za górami za lasami żył sobie piękna dziewczynka, '
        start = [0]
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)
        for i in range(num_words_to_generate):      
            input_vectors_padded = self._preprocess_input_text(generated_text)
            output = tf.transpose(output_array.stack())
            predictions = self.model.predict([input_vectors_padded, output], verbose = 0)[0]
            predictions = predictions[-1:, :]
            predicted_id = np.argmax(predictions)
            output_array = output_array.write(i+1, [predicted_id])
            predicted_word = TOKENIZER.index_word[predicted_id]
            generated_text += " " + predicted_word

        print()
        print('Epoch:',  (epoch + 1))
        print('Prediction:', generated_text)


"""BASIC TRAIN FUNCTION TO SHOW THAT IT CAN BE TRAINED"""
def train():
    # hyper parameters for transformer model
    num_layers = 2
    emb_dim = 300
    feed_forward = 256
    num_heads = 4
    dropout_rate = 0.2

    transformer = Transformer(
        num_layers = num_layers, 
        emb_dim = emb_dim, 
        num_heads = num_heads, 
        feed_forward = feed_forward, 
        input_vocab_size = len(TOKENIZER.word_index) + 1, 
        target_vocab_size = len(TOKENIZER.word_index) + 1,
        dropout_rate = dropout_rate)

    transformer.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')
    transformer.fit([TARGETS, SEQUENCES], TARGETS,  batch_size=64, epochs=40, callbacks=[PredictionCallback()])


if __name__ == "__main__":
    train()