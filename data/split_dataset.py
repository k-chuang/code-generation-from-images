import sys

sys.path.extend(['..'])

import os
import numpy as np
import shutil
import hashlib


def split_dataset(input_path):
    '''
    Split dataset into train, eval, and test sets
    :param input_path: path to location of all images and associated gui code
    :return: None
    '''
    distribution = 6 
    TRAINING_SET_NAME = "img/train_images"
    EVALUATION_SET_NAME = "img/eval_images"
    TESTING_SET_NAME = "img/test_images"

    paths = []
    for f in os.listdir(input_path):
        if f.find(".gui") != -1:
            path_gui = "{}/{}".format(input_path, f)
            file_name = f[:f.find(".gui")]

            if os.path.isfile("{}/{}.png".format(input_path, file_name)):
                path_img = "{}/{}.png".format(input_path, file_name)
                paths.append(file_name)
    
    
    testing_samples_number = len(paths) / (distribution + 1)
    training_samples_number = (testing_samples_number * distribution) - 100
    evaluation_samples_number = 100

    assert training_samples_number + evaluation_samples_number + testing_samples_number == len(paths)

    print("Splitting datasets, training samples: {}, testing samples: {}, evaluation samples: {}".format(int(training_samples_number),
                                                                                                         int(testing_samples_number),
                                                                                                         int(evaluation_samples_number)))

    np.random.shuffle(paths)

    eval_set = []
    train_set = []
    test_set = []
    hashes = []
    for path in paths:
        if sys.version_info >= (3,):
            f = open("{}/{}.gui".format(input_path, path), 'r', encoding='utf-8')
        else:
            f = open("{}/{}.gui".format(input_path, path), 'r')

        with f:
            chars = ""
            for line in f:
                chars += line
            content_hash = chars.replace(" ", "").replace("\n", "")
            content_hash = hashlib.sha256(content_hash.encode('utf-8')).hexdigest()

            if len(eval_set) == evaluation_samples_number and len(test_set) == testing_samples_number:
                train_set.append(path)
            else:
                is_unique = True
                for h in hashes:
                    if h is content_hash:
                        is_unique = False
                        break

                if is_unique and len(test_set) != testing_samples_number:
                    test_set.append(path)
                elif is_unique and len(eval_set) != evaluation_samples_number:
                    eval_set.append(path)
                else:
                    train_set.append(path)
                

            hashes.append(content_hash)
    assert len(test_set) == testing_samples_number
    assert len(eval_set) == evaluation_samples_number
    assert len(train_set) == training_samples_number

    if not os.path.exists("{}/{}".format(os.path.dirname(input_path), TESTING_SET_NAME)):
        os.makedirs("{}/{}".format(os.path.dirname(input_path), TESTING_SET_NAME))
    
    if not os.path.exists("{}/{}".format(os.path.dirname(input_path), EVALUATION_SET_NAME)):
        os.makedirs("{}/{}".format(os.path.dirname(input_path), EVALUATION_SET_NAME))

    if not os.path.exists("{}/{}".format(os.path.dirname(input_path), TRAINING_SET_NAME)):
        os.makedirs("{}/{}".format(os.path.dirname(input_path), TRAINING_SET_NAME))
    
    for path in test_set:
        shutil.copyfile("{}/{}.png".format(input_path, path), "{}/{}/{}.png".format(os.path.dirname(input_path), TESTING_SET_NAME, path))
        shutil.copyfile("{}/{}.gui".format(input_path, path), "{}/{}/{}.gui".format(os.path.dirname(input_path), TESTING_SET_NAME, path))

    for path in eval_set:
        shutil.copyfile("{}/{}.png".format(input_path, path), "{}/{}/{}.png".format(os.path.dirname(input_path), EVALUATION_SET_NAME, path))
        shutil.copyfile("{}/{}.gui".format(input_path, path), "{}/{}/{}.gui".format(os.path.dirname(input_path), EVALUATION_SET_NAME, path))

    for path in train_set:
        shutil.copyfile("{}/{}.png".format(input_path, path), "{}/{}/{}.png".format(os.path.dirname(input_path), TRAINING_SET_NAME, path))
        shutil.copyfile("{}/{}.gui".format(input_path, path), "{}/{}/{}.gui".format(os.path.dirname(input_path), TRAINING_SET_NAME, path))

    print("Training dataset: {}/training_set".format(os.path.dirname(input_path), path))
    print("Evaluation dataset: {}/eval_set".format(os.path.dirname(input_path), path))
    print("Testing dataset: {}/test_set".format(os.path.dirname(input_path), path))


def partition_data(input_path):
    '''
    Partitions the data if it is not already split
    :param input_path: path to location of all images and associated gui code
    :return: None
    '''
    if not os.path.exists('../data/img/train_images'):
        split_dataset(input_path)
    else:
        print('Training set images already exist at ../data/img/train_images and are ready to be converted to arrays')
        
    if os.path.exists('../data/img/eval_images'):
        print('Evaluation set images already exist at ../data/img/eval_images and are ready to be converted to arrays')

    if os.path.exists('../data/img/test_images'):
        print('Test set images already exist at ../data/img/test_images and are ready to be converted to arrays')
