from generator import GenerateStories
from helpers.functions import unpickle
from transformer import Transformer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def user_input_string():
    while True:
        user_prompt = input()
        try:
            user_prompt = str(user_prompt)
            break

        except ValueError:
            print("Wprowadziłeś nieprawidłowe dane")
            continue

    return user_prompt


def user_input_int():
    while True:
        user_word_to_generate= input()
        try:
            user_word_to_generate = int(user_word_to_generate)
            break

        except ValueError:
            print("Wprowadziłeś nieprawidłowe dane")
            continue

    return user_word_to_generate


def main():
    tokenizer = unpickle('data/tokenizer.pkl')
    targets = unpickle('data/targets.pkl')
    sequences = unpickle('data/sequences.pkl')
    name = 'Transformer_final.h5'
    
    num_layers = 2
    emb_dim = 300
    feed_forward = 256
    num_heads = 4
    dropout_rate = 0.2

    model = Transformer(
        num_layers = num_layers, 
        emb_dim = emb_dim, 
        num_heads = num_heads, 
        feed_forward = feed_forward, 
        input_vocab_size = len(tokenizer.word_index) + 1, 
        target_vocab_size = len(tokenizer.word_index) + 1,
        dropout_rate = dropout_rate)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')
    model.fit([targets[:128], sequences[:128]], targets[:128],  batch_size=64, epochs=1)
    model.load_weights('models/' + 'Weights_' + name)
    
    print("Wpisz liczbę słów do wygenerowania przez transformer: ")
    word_to_generate = user_input_int()
    print("Wpisz prompt: ")
    promts = [user_input_string()]
    print("MOGĄ POJAWIĆ SIĘ BŁĘDY TENSORFLOW, PROSZĘ SIĘ NIE PRZEJMOWAĆ!")
    temperatures = [0.2, 0.5, 1.0]
    stories = GenerateStories(model, tokenizer)
    stories.generate_story(promts, word_to_generate, temperatures)
    stories.show_stories()


if __name__ == "__main__":
    main()