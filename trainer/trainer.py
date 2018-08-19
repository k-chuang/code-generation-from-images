import sys
sys.path.extend(['..'])
from config.config import *

from keras import backend as K
K.set_image_data_format(IMAGE_FILE_FORMAT)

import tensorflow as tf
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from data.load_data import load_data, load_doc
from keras.preprocessing.text import Tokenizer
from model.CodeGeneratorModel import CodeGeneratorModel
from data_loader.DataGenerator import *
from utils.callbacks import *
from contextlib import redirect_stdout


def trainer(train_dir_name, eval_dir_name, out_dir_name):
    '''
    Train the model
    :param train_dir_name: path to training set directory
    :param eval_dir_name: path to evaluation set directory
    :param out_dir_name: output path to save model files
    :return: None
    '''
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)

    train_features, train_texts = load_data(train_dir_name)
    eval_features, eval_texts = load_data(eval_dir_name)
    steps_per_epoch = len(train_texts) / BATCH_SIZE
    print('Image file format is %s' % IMAGE_FILE_FORMAT)
    print('Keras backend file format is %s' % K.image_data_format())
    print('Training images input shape: {}'.format(train_features.shape))
    print('Evaluation images input shape: {}'.format(eval_features.shape))
    print('Training texts shape: {}'.format(len(train_texts)))
    print('Evaluation texts input shape: {}'.format(len(eval_texts)))
    print('Epoch size: {}'.format(EPOCHS))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Steps per epoch: {}'.format(int(steps_per_epoch)))
    print('Kernel Initializer: {}'.format(KERNEL_INIT))

    with open(os.path.join(out_dir_name, 'config.txt'), 'w') as fh:
        with redirect_stdout(fh):
            print('Image file format is %s' % IMAGE_FILE_FORMAT)
            print('Keras backend file format is %s' % K.image_data_format())
            print('Training images input shape: {}'.format(train_features.shape))
            print('Evaluation images input shape: {}'.format(eval_features.shape))
            print('Training texts shape: {}'.format(len(train_texts)))
            print('Evaluation texts input shape: {}'.format(len(eval_texts)))
            print('Epoch size: {}'.format(EPOCHS))
            print('Batch size: {}'.format(BATCH_SIZE))
            print('Steps per epoch: {}'.format(int(steps_per_epoch)))
            print('Kernel Initializer: {}'.format(KERNEL_INIT))

    # Prepare tokenizer to create the vocabulary
    tokenizer = Tokenizer(filters='', split=" ", lower=False)
    # Create the vocabulary
    tokenizer.fit_on_texts([load_doc('../data/code.vocab')])

    # Initialize data generators for training and validation
    train_generator = DataGenerator(train_texts, train_features, batch_size=BATCH_SIZE,
                                    tokenizer=tokenizer, shuffle=True, image_data_format=IMAGE_FILE_FORMAT)
    validation_generator = DataGenerator(eval_texts, eval_features, batch_size=BATCH_SIZE,
                                         tokenizer=tokenizer, shuffle=True, image_data_format=IMAGE_FILE_FORMAT)

    # Initialize model
    model = CodeGeneratorModel(IMAGE_SIZE, out_dir_name, image_file_format=IMAGE_FILE_FORMAT, kernel_initializer=KERNEL_INIT)
    model.save_model()
    model.summarize()
    model.summarize_image_model()
    model.plot_model()

    if VALIDATE:
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=generate_callbacks(out_dir_name),
                            validation_data=validation_generator)
    else:
        model.fit_generator(generator=train_generator,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=generate_callbacks(out_dir_name))


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) != 1:
        print('Need to supply an argument specifying output path')
        exit(0)
    out_dir = argv[0]
    train_dir = '../data/train/'
    eval_dir = '../data/eval/'
    # out_dir = '../results/exp9'
    trainer(train_dir, eval_dir, out_dir)
