import pickle

def unpickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)