import keras
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from config.config import *


class DataGenerator(keras.utils.Sequence):
    # initialization
    def __init__(self, text_sequences, image_features, batch_size, tokenizer, image_data_format, shuffle=True):
        self.text_sequences = text_sequences
        self.image_features = image_features
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.image_data_format = image_data_format
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.text_sequences) / self.batch_size))

    def __getitem__(self, index):
        '''Dunder/Magic method that generates one batch of data, is called when using DataGenerator[index]'''
        # Generate indexes of the batch (self.indexes has already been randomized)
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        batch_text_sequences = [self.text_sequences[k] for k in indexes]
        batch_image_features = [self.image_features[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_text_sequences, batch_image_features,
                                      CONTEXT_LENGTH, image_file_format=self.image_data_format)

        return X, y

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.text_sequences))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @staticmethod
    def preprocess_data(texts, features, max_sequence, tokenizer):
        '''
        Preprocess each individual image & sequence pair by generating time series of the whole sequence token by token
        :param texts: one list of texts containing the DSL GUI tokens associated with a webpage
        :param features: an numpy array of the image (256, 256, 3)
        :param max_sequence: maximum length a sequence can reach
        :param tokenizer: a tokenizer with knowledge and mapping of datasets vocabulary
        :return: image_data = (# of tokens/samples in sequence, 256, 256, 3)
                 X = (# of tokens/samples in sequence, context_length=48)
                 y= (# of tokens/samples in sequence, context_length=48)
        '''
        X, y, image_data = list(), list(), list()
        sequences = tokenizer.texts_to_sequences(texts)
        # Loop through the length of the the specific sequence (different for each gui file)
        for img_no, seq in enumerate(sequences):
            # Create a image and sequence pairs by converting sequence to a (time) series of sequences
            # And repeating the image to the length of the sequence
            for i in range(1, len(seq)):
                # Add the sentence until the current count(i) and add the current count to the output
                in_seq, out_seq = seq[:i], seq[i]
                # Pad all the input token sentences to max_sequence
                in_seq = pad_sequences([in_seq], maxlen=max_sequence, dtype='int8')[0]
                # Turn the output into one-hot encoding
                out_seq = to_categorical([out_seq], num_classes=VOCAB_SIZE)[0].astype('int8')
                # Add the corresponding image to the bootstrap token file
                image_data.append(features[img_no])
                # Cap the input sentence to 48 tokens and add it
                X.append(in_seq[-48:])
                y.append(out_seq)
        return np.array(image_data), np.array(X), np.array(y)

    def __data_generation(self, batched_text_sequences, batched_image_features, max_sequence, image_file_format='channels_last'):
        '''
        Generate batches of data (sequence features & image features pair)
        :param batched_text_sequences: a batch of the train texts/sequences for each website image (# of training samples,)
        :param batched_image_features: a batch of numpy arrays representing each image (# of training samples, 256, 256, 3)
        :param max_sequence: maximum length a sequence can reach
        :return: list of numpy arrays with first item as the pair of sequence features & image features and second
        item as label
        '''
        # initialization
        x_imgs, x_texts, y = list(), list(), list()
        for j in range(0, self.batch_size):
            image = batched_image_features[j]
            # retrieve text input
            desc = batched_text_sequences[j]
            # generate input-output pairs
            in_img, in_seq, out_word = self.preprocess_data([desc], [image], max_sequence, self.tokenizer)
            for k in range(len(in_img)):
                x_imgs.append(in_img[k])
                x_texts.append(in_seq[k])
                y.append(out_word[k])
        if image_file_format == 'channels_first':
            return [[np.transpose(np.array(x_imgs), [0, 3, 1, 2]), np.array(x_texts)], np.array(y)]
        else:
            return [[np.array(x_imgs), np.array(x_texts)], np.array(y)]

