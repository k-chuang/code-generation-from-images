import sys
sys.path.extend(['..'])

import tensorflow as tf
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from generator.generate_code import *
from nltk.translate.bleu_score import corpus_bleu
from config.config import *
from base.BaseModel import *
from utils.tokenizer import *


def evaluate_model(input_path, model_path, tokenizer, max_length=48, display=False):
    '''
    Evaluate model by comparing actual vs predictions via the BLEU scoring criteria
    :param input_path: input path containing images + gui code pairs to evaluate model on
    :param model_path: path to model files
    :param tokenizer: a Keras Tokenizer object fit on vocab
    :param max_length: context length
    :param display: bool on whether to print out DSL code predictions and actual labels to standard output
    :return: 4-ngram BLEU score, list of actual DSL code, list of predicted DSL code
    '''
    model_json_path = glob.glob(os.path.join(model_path, '*.json'))[0]
    model_weights_path = glob.glob(os.path.join(model_path, '*.hdf5'))[0]
    with open(model_json_path, 'r') as fh:
        model_json = fh.read()
    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    print('Successfully loaded model and model weights...')

    images, texts = load_data(input_path)
    actual, predictions = list(), list()
    for i in range(len(texts)):
        predicted_code = generate_code(model, images[i], tokenizer, max_length, display)
        # store actual and predicted
        if display:
            print('\n\nActual---->\n\n' + texts[i])
        actual.append([texts[i].split()])
        predictions.append(predicted_code.split())
    bleu = corpus_bleu(actual, predictions)
    return bleu, actual, predictions


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) != 1:
        print('Need to supply an argument specifying model path')
        exit(0)
    model_path = argv[0]
    test_dir = '../data/test/'
    # model_path = '../results/'
    vocab_path = '../data/code.vocab'

    tokenizer = tokenizer(vocab_path)
    bleu, actual, predictions = evaluate_model(test_dir, model_path, tokenizer, CONTEXT_LENGTH, display=False)
    # Calculate BLEU score (standard is 4-gram, but just get all individual N-Gram BLEU scores from 1 gram to 4 gram)
    # By default, the sentence_bleu() and corpus_bleu() scores calculate the cumulative 4-gram BLEU score, also called BLEU-4.
    # It is common to report the cumulative BLEU-1 to BLEU-4 scores when describing the skill of a text generation system.
    # 4-gram is the most strict and corresponds the best to human translations
    print('BLEU-1: %f' % corpus_bleu(actual, predictions, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predictions, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predictions, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predictions, weights=(0.25, 0.25, 0.25, 0.25)))

    bleu_score_path = os.path.join(model_path, 'bleu_score.txt')
    with open(bleu_score_path, 'w') as fh:
        fh.write('Test set dir: %s\n' % test_dir)
        fh.write('BLEU-1: %f \n' % corpus_bleu(actual, predictions, weights=(1.0, 0, 0, 0)))
        fh.write('BLEU-2: %f \n' % corpus_bleu(actual, predictions, weights=(0.5, 0.5, 0, 0)))
        fh.write('BLEU-3: %f \n' % corpus_bleu(actual, predictions, weights=(0.3, 0.3, 0.3, 0)))
        fh.write('BLEU-4: %f \n' % corpus_bleu(actual, predictions, weights=(0.25, 0.25, 0.25, 0.25)))


