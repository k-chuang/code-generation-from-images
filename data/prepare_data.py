import cv2
import os
import numpy as np
import shutil
from data.split_dataset import split_dataset, partition_data


def get_preprocessed_image(img_path, image_size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype('float32')
    img /= 255
    return img


def convert_image_to_array(input_path, output_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    IMAGE_SIZE = 256
    print("Converting images to numpy arrays...")
    for f in os.listdir(input_path):
        if f.find(".png") != -1:
            img = get_preprocessed_image("{}/{}".format(input_path, f), IMAGE_SIZE)
            file_name = f[:f.find(".png")]
            np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
            retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

            assert np.array_equal(img, retrieve)

            shutil.copyfile("{}/{}.gui".format(input_path, file_name), "{}/{}.gui".format(output_path, file_name))

    print("Numpy arrays saved in {}".format(output_path))


def prepare_data(train_dir_name, eval_dir_name, test_dir_name):
    if not os.path.exists(train_dir_name):
        os.makedirs(train_dir_name)
        convert_image_to_array('../data/img/train_images', train_dir_name)
    else:
        print('Training set already exists at %s' % train_dir_name)

    if not os.path.exists(eval_dir_name):
        os.makedirs(eval_dir_name)
        convert_image_to_array('../data/img/eval_images', eval_dir_name)
    else:
         print('Evaluation set already exists at %s' % eval_dir_name)

    if not os.path.exists(test_dir_name):
        os.makedirs(test_dir_name)
        convert_image_to_array('../data/img/test_images', test_dir_name)
    else:
         print('Test set already exists at %s' % test_dir_name)

    print('All data successfully converted to numpy arrays...')


def main():
    train_dir_name = '../data/train/'
    test_dir_name = '../data/test/'
    eval_dir_name = '../data/eval/'
    # Preparing data
    partition_data('../data/all_data')
    prepare_data(train_dir_name, eval_dir_name, test_dir_name)

if __name__ == '__main__':
    main()
