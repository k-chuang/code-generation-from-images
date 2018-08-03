
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import sys, os
import numpy as np
from data.load_data import load_doc

sys.path.append('..')    


def preprocess_data(texts, features, max_sequence):
    X, y, image_data = list(), list(), list()
    # Initialize the function to create the vocabulary
    tokenizer = Tokenizer(filters='', split=" ", lower=False)
    # Create the vocabulary
    tokenizer.fit_on_texts([load_doc('../data/bootstrap.vocab')])
    # Add one spot for the empty word in the vocabulary (17 vocabulary words + 1 = 18 (vocab_size))
    vocab_size = len(tokenizer.word_index) + 1
    # max_length = 48
    sequences = tokenizer.texts_to_sequences(texts)
    # Loop through the length of the the specific sequence
    for img_no, seq in enumerate(sequences):
        # Create a image and sequence pairs by converting sequence to a (time) series of sequences
        # And repeating the image to the length of the sequence
        for i in range(1, len(seq)):
            # Add the sentence until the current count(i) and add the current count to the output
            in_seq, out_seq = seq[:i], seq[i]
            # Pad the input sequences with 0's to max_sequence (out -> 150 tokens)
            in_seq = pad_sequences([in_seq], maxlen=max_sequence, dtype='int8')[0]
            # Turn the output into one-hot encoding
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0].astype('int8')
            # Add the corresponding image to the boostrap token file
            image_data.append(features[img_no])
            # Cap the input sentence to the LAST 48 tokens and append it to the sequences
            X.append(in_seq[-48:])
            y.append(out_seq)
    # image_data = (# of tokens/samples in sequence, 256, 256, 3)
    # X = (# of tokens/samples in sequence, vocab_size=48)
    # y= (# of tokens/samples in sequence, vocab_size=48)
    return np.array(image_data), np.array(X), np.array(y)

# data generator, intended to be used in a call to model.fit_generator()
# This will save memory and prevent MemoryErrors from numpy
def data_generator(descriptions, features, n_step, max_sequence):
    # loop until we finish training
    while 1:
        # loop over photo identifiers in the dataset
        for i in range(0, len(descriptions), n_step):
            Ximages, XSeq, y = list(), list(), list()
            for j in range(i, min(len(descriptions), i+n_step)):
                # Get image input
                image = features[j]
                # Get text input
                desc = descriptions[j]
                # generate input-output pairs
                in_img, in_seq, out_word = preprocess_data([desc], [image], max_sequence)
                # Output of above:
                # in_img = (# of tokens/samples in sequence, 256, 256, 3)
                # in_seq = (# of tokens/samples in sequence, vocab_size=48)
                # out_word = (# of tokens/samples in sequence, vocab_size=48)
                for k in range(len(in_img)):
                    Ximages.append(in_img[k])
                    XSeq.append(in_seq[k])
                    y.append(out_word[k])

            # yield this batch of samples to the model
            # yield [[in_img, in_seq], out_word ]
            yield [[np.array(Ximages), np.array(XSeq)], np.array(y)]
#             yield [[np.transpose(np.array(Ximages), [0, 3, 1, 2]), np.array(XSeq)], np.array(y)]