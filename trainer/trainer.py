import sys
sys.path.extend(['..'])

from utils.memory_saving_gradients import *

# from tensorflow.python.keras._impl.keras import backend as K
# K.__dict__["gradients"] = gradients_memory

import tensorflow as tf
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from data.prepare_data import prepare_data
from data.split_dataset import partition_data
from data.load_data import load_data, load_doc
from keras.preprocessing.text import Tokenizer
from model.CodeGeneratorModel import CodeGeneratorModel
from data_loader.DataGenerator import *
from utils.callbacks import *
from config.config import *


def trainer():

    train_dir_name = '../data/train/'
    test_dir_name = '../data/test/'
    eval_dir_name = '../data/eval/'

    outD = '../results/'
    if not os.path.exists(outD):
        os.makedirs(outD)

    # Preparing data
    partition_data('../data/all_data')
    prepare_data(train_dir_name, eval_dir_name, test_dir_name)

    train_features, train_texts = load_data(train_dir_name)
    test_features, test_texts = load_data(test_dir_name)
    eval_features, eval_texts = load_data(eval_dir_name)

    steps_per_epoch = len(train_texts) / BATCH_SIZE

    # Prepare tokenizer to create the vocabulary
    tokenizer = Tokenizer(filters='', split=" ", lower=False)
    # Create the vocabulary
    tokenizer.fit_on_texts([load_doc('../data/bootstrap.vocab')])

    # Initialize data generator
    train_generator = DataGenerator(train_texts, train_features, batch_size=1, tokenizer=tokenizer, shuffle=True)
    validation_generator = DataGenerator(eval_texts, eval_features, batch_size=1, tokenizer=tokenizer, shuffle=True)

    # Initialize model
    model = CodeGeneratorModel(IMAGE_SIZE, outD)

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=generate_callbacks(outD),
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6)
    # x, y = data_loader[0]
    # data_loader = data_generator(train_texts, train_features, 2, 150)

if __name__ == '__main__':
    trainer()
