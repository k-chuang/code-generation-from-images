import sys

sys.path.extend(['..'])

import tensorflow as tf

from data.prepare_data import prepare_data
from data.split_dataset import partition_data
from data.load_data import load_data
from data_loader.data_generator import *
from models.code_generator_model import CodeGeneratorModel
from trainers.trainer import CodeGeneratorTrainer

from utils.config import process_config
from utils.logger import DefinedSummarizer, Logger
from utils.dirs import create_dirs


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    # try:
    #     args = get_args()
    #     config = process_config(args.config)

    # except:
    #     print("missing or invalid arguments")
    #     exit(0)

    train_dir_name = '../data/train/'
    test_dir_name = '../data/test/'
    eval_dir_name = '../data/eval/'

    # Preparing data
    partition_data('../data/all_data')
    prepare_data(train_dir_name, eval_dir_name, test_dir_name)

    train_features, train_texts = load_data(train_dir_name)
    test_features, test_texts = load_data(test_dir_name)
    eval_features, eval_texts = load_data(eval_dir_name)

    config = process_config('../config/config.json')

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # # create tensorflow session
    # sess = tf.Session()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        # create your data generator
        data_loader = data_generator(train_texts, train_features, 1, 150)

        # # create instance of the model you want
        model = CodeGeneratorModel(config, is_training=True)

        # # create tensorboard logger
        logger = DefinedSummarizer(sess, summary_dir=config.summary_dir,
                                   scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
                                                'test/loss_per_epoch','test/acc_per_epoch'])

        # create trainer and path all previous components to it
        trainer = CodeGeneratorTrainer(sess, model, config, logger, data_loader)
        # here you train your model
        trainer.train()


if __name__ == '__main__':
    main()