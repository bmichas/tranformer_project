from generator import GenerateStories
from helpers.functions import unpickle
from transformer import Transformer
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
    name = 'Transformer_final32'
    model = load_model('models/' + name + '.keras')
    model.load_weights('models/' + 'Weights_' + name + '.keras')
    
    print("Wpisz liczbę słów do wygenerowania przez transformer: ")
    word_to_generate = user_input_int()
    print("Wpisz prompt: ")
    promts = [user_input_string()]

    temperatures = [0.2, 0.5, 1.0]
    stories = GenerateStories(model, tokenizer)
    stories.generate_story(promts, word_to_generate, temperatures)
    stories.show_stories()


if __name__ == "__main__":
    main()