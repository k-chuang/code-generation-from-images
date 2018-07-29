from tqdm import tqdm
import tensorflow as tf
import numpy as np


class DataGenerator:
    """
    Manual Loading
    Using placeholders and python generators
    """

    def __init__(self, config):
        self.config = config

        self.x_train = np.load(config.x_train)
        self.y_train = np.load(config.y_train)
        self.x_test = np.load(config.x_test)
        self.y_test = np.load(config.y_test)

        print("x_train shape: {} dtype: {}".format(self.x_train.shape, self.x_train.dtype))
        print("y_train shape: {} dtype: {}".format(self.y_train.shape, self.y_train.dtype))
        print("x_test shape: {} dtype: {}".format(self.x_test.shape, self.x_test.dtype))
        print("y_test shape: {} dtype: {}".format(self.y_test.shape, self.y_test.dtype))

        self.len_x_train = self.x_train.shape[0]
        self.len_x_test = self.x_test.shape[0]

        self.num_iterations_train = self.len_x_train // self.config.batch_size
        self.num_iterations_test = self.len_x_test // self.config.batch_size

    def get_input(self):
        x = tf.placeholder(tf.float32, [None, self.config.img_h, self.config.img_w, 3])
        y = tf.placeholder(tf.int64, [None, ])

        return x, y

    def generator_train(self):
        start = 0
        idx = np.random.choice(self.len_x_train, self.len_x_train, replace=False)
        while True:
            mask = idx[start:start + self.config.batch_size]
            x_batch = self.x_train[mask]
            y_batch = self.y_train[mask]

            start += self.config.batch_size

            yield x_batch, y_batch

            if start >= self.len_x_train:
                return

    def generator_test(self):
        start = 0
        idx = np.random.choice(self.len_x_test, self.len_x_test, replace=False)
        while True:
            mask = idx[start:start + self.config.batch_size]
            x_batch = self.x_test[mask]
            y_batch = self.y_test[mask]

            start += self.config.batch_size

            yield x_batch, y_batch

            if start >= self.len_x_test:
                return
