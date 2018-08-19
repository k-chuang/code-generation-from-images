import sys

sys.path.extend(['..'])

import tensorflow as tf

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras.preprocessing.sequence import pad_sequences
from config.config import *
from base.BaseModel import *
from utils.tokenizer import *
from data.prepare_data import get_preprocessed_image
from compiler.classes.Compiler import *


def generate_code(model, image, tokenizer, max_length=48, display=False):
    '''
    Generate predictions of DSL code
    :param model: neural network model (architecture + weights) to use for generating
    :param image: specific image to generate code for
    :param tokenizer: a Keras Tokenizer object fit on vocab
    :param max_length: context length
    :param display: bool on whether to print out DSL code predictions to standard output
    :return: generated code for a specific image
    '''
    image = np.array([image])
    # seed the generation process
    generated_code = '<START> '
    # iterate over the whole length of the sequence
    if display:
        print('\nPrediction---->\n\n<START> ', end='')
    for i in range(150):
        # integer encode input sequence (Produces list of unique integers)
        sequence = tokenizer.texts_to_sequences([generated_code])[0]
        # pad sequence to reach max_length (Pads 0's before sequence to reach 48)
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # Get the index of the prediction with highest probability
        yhat = np.argmax(yhat)
        # Get word based on unique integer with max probability
        word = get_word(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        generated_code += word + ' '
        # stop if we predict the end of the sequence
        if display:
            print(word + ' ', end='')
        if word == '<END>':
            break
    return generated_code


def generate_dsl(input_path, dsl_dir, model_path, tokenizer, max_length=48, write=False, display=False):
    '''
    Generate DSL code
    :param input_path: input directory containing png images
    :param dsl_dir: output directory to save .gui files
    :param model_path: path to model (architecture and weights)
    :param tokenizer: a Keras Tokenizer object fit on vocab
    :param max_length: context length
    :param write: bool on whether to write DSL code to file
    :param display: bool on whether to print out DSL code predictions to standard output
    :return: list of generated dsl code predictions
    '''
    dsl_predictions = []
    model_json_path = glob.glob(os.path.join(model_path, '*.json'))[0]
    model_weights_path = glob.glob(os.path.join(model_path, '*.hdf5'))[0]
    with open(model_json_path, 'r') as fh:
        model_json = fh.read()
    model = model_from_json(model_json)
    model.load_weights(model_weights_path)
    print('Successfully loaded model and model weights...')

    for f in glob.glob(os.path.join(input_path, '*.png')):
        img = get_preprocessed_image(f, IMAGE_SIZE[0])
        prediction = generate_code(model, img, tokenizer, max_length, display)
        dsl_predictions.append(prediction)
        fname = f[:f.find(".png")]
        if write:
            out_name = os.path.join(dsl_dir, os.path.basename(fname)) + '.gui'
            with open(out_name, 'w') as fh:
                fh.write(prediction)
        print('\n\nGenerated DSL code for %s...\n' % os.path.basename(f))

    return dsl_predictions

def generate_html(input_dir, dsl_mapping, output_dir):
    '''
    Generate html code from compiling DSL code
    :param input_dir: directory containing all DSL file codes (gui files)
    :param dsl_mapping: dsl mapping to html associated code
    :param output_dir: directory to save html files
    :return: list of compiled websites
    '''
    compiled_websites = []

    for gui in glob.glob(os.path.join(input_dir, '*.gui')):
        compiler = Compiler(dsl_mapping)
        fname = os.path.basename(gui[:gui.find(".gui")])
        with open(gui, 'r') as fh:
            dsl_code = fh.read()
        dsl_code = dsl_code.split()
        out_name = os.path.join(output_dir, fname) + '.html'
        compiled_webpage = compiler.compile(dsl_code, out_name)
        print('Saved html file at %s' % out_name)
        compiled_websites.append(compiled_webpage)

    return compiled_websites


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) < 2:
        print('Two arguments are required...')
        print('Usage: generate_code.py <image path> <model path>')
        exit(0)
    test_img_dir = argv[0]
    model_path = argv[1]

    # test_img_dir = '../data/img/test_images'
    # model_path = '../results/'
    vocab_path = '../data/code.vocab'
    dsl_dir = os.path.join(model_path, 'generated_dsl')
    if not os.path.exists(dsl_dir):
        print('Generating DSL code...')
        os.makedirs(dsl_dir)
        generate_dsl(test_img_dir, dsl_dir, model_path, tokenizer(vocab_path), CONTEXT_LENGTH, write=True, display=False)
    else:
        print('DSL directory already exists...')

    dsl_mapping = '../compiler/assets/web-dsl-mapping.json'
    html_dir = os.path.join(model_path, 'generated_html')
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
        print('Compiling DSL to HTML code...')
        compiled_websites = generate_html(dsl_dir, dsl_mapping, html_dir)
    else:
        print('HTML directory already exists...')




