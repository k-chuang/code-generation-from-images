import os
import numpy as np


# Read a file and return a string
def load_doc(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text


def load_data(data_dir):
    text = []
    images = []
    # Load all the files and order them
    all_filenames = os.listdir(data_dir)
    all_filenames.sort()
    for filename in (all_filenames):
        if filename[-3:] == "npz":
            # Load the images already prepared in arrays
            image = np.load(data_dir+filename)
            images.append(image['features'])
        else:
            # Load the boostrap tokens and rap them in a start and end tag
            syntax = '<START> ' + load_doc(data_dir+filename) + ' <END>'
            # Separate all the words with a single space
            syntax = ' '.join(syntax.split())
            # Add a space after each comma
            syntax = syntax.replace(',', ' ,')
            text.append(syntax)
    images = np.array(images, dtype=np.float32)

    # if image_data_format == 'channels_first':
    #     images = tf.transpose(images, [0, 3, 1, 2])

    return images, text
