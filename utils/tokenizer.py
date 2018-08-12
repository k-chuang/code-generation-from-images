from keras.preprocessing.text import Tokenizer
from data.load_data import *


def get_word(integer, tokenizer):
    '''
    Find a word given the unique integer and a tokenizer, else return None
    :param integer:
    :param tokenizer:
    :return:
    '''
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def tokenizer(vocab_path):
    # Prepare tokenizer to create the vocabulary
    tokenizer = Tokenizer(filters='', split=" ", lower=False)
    # Create the vocabulary
    tokenizer.fit_on_texts([load_doc(vocab_path)])
    return tokenizer