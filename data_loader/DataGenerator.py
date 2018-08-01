import keras
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

class DataGenerator(keras.utils.Sequence):
    # initialization
    def __init__(self, train_texts, train_features, batch_size, tokenizer, shuffle=True):
        self.train_texts = train_texts
        self.train_features = train_features
        #         self.dim_x = dim_x
        #         self.dim_y = dim_y
        #         self.list_IDs = list_IDs
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.train_texts) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        batch_train_texts = [self.train_texts[k] for k in indexes]
        batch_train_features = [self.train_features[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_train_texts, batch_train_features, 150)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.train_texts))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def preprocess_data(texts, features, max_sequence, tokenizer):
        X, y, image_data = list(), list(), list()
        sequences = tokenizer.texts_to_sequences(texts)
        for img_no, seq in enumerate(sequences):
            for i in range(1, len(seq)):
                # Add the sentence until the current count(i) and add the current count to the output
                in_seq, out_seq = seq[:i], seq[i]
                # Pad all the input token sentences to max_sequence
                in_seq = pad_sequences([in_seq], maxlen=max_sequence)[0]
                # Turn the output into one-hot encoding
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # Add the corresponding image to the boostrap token file
                image_data.append(features[img_no])
                # Cap the input sentence to 48 tokens and add it
                X.append(in_seq[-48:])
                y.append(out_seq)
        return np.array(image_data), np.array(X), np.array(y)

    def __data_generation(self, train_texts, train_images, max_sequence):
        # initialization
        #         X1 = np.empty((self.batch_size, self.dim_x, self.dim_y, 3))
        #         X2 = np.empty((self.batch_size, self.number_features))
        #         Y = np.empty((self.batch_size), dtype = int)

        for i in range(0, len(train_texts), self.batch_size):
            Ximages, XSeq, y = list(), list(), list()
            for j in range(i, min(len(train_texts), i + self.batch_size)):
                image = train_images[j]
                # retrieve text input
                desc = train_texts[j]
                # generate input-output pairs
                in_img, in_seq, out_word = self.preprocess_data([desc], [image], max_sequence, self.tokenizer)
                for k in range(len(in_img)):
                    Ximages.append(in_img[k])
                    XSeq.append(in_seq[k])
                    y.append(out_word[k])

        return [[np.array(Ximages), np.array(XSeq)], np.array(y)]

