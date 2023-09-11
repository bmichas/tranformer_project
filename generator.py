import numpy as np
import tensorflow as tf


class GenerateStories:
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer


    def preprocess_input_text(self, text):
        text = text.split()
        sequence = text
        vector_sequences = [[word for word in sequence]]
        vector_sequences = self.tokenizer.texts_to_sequences(vector_sequences)
        return tf.convert_to_tensor(vector_sequences)


    def sample(self, preds, temperature):
        preds = np.asarray(preds[-1]).astype("float64")
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    def generate_story(self, promts, lenghth, temperatures):
        self.generated_stories = []
        for promt in promts:
            for temperature in temperatures:
                generated_text = promt
                start = [0]
                output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
                output_array = output_array.write(0, start)
                for i in range(lenghth):      
                    input_vectors_padded = self.preprocess_input_text(generated_text)
                    output = tf.transpose(output_array.stack())
                    predictions = self.model.predict([input_vectors_padded, output], verbose = 0)[0]
                    predictions = predictions[-1:, :]
                    predicted_id = self.sample(predictions, temperature)
                    output_array = output_array.write(i+1, [predicted_id])
                    predicted_word = self.tokenizer.index_word[predicted_id]
                    generated_text += " " + predicted_word

                self.generated_stories.append([temperature, promt, generated_text])

    def show_stories(self):
        for story in self.generated_stories:
            print()
            print('Temperature:',  story[0])
            print('Prompt:', story[1])
            print('Prediction:', story[2])