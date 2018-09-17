# Code Generation from Images

An end-to-end deep neural network designed in Keras (with tensorflow backend) that will transform a screenshot into Bootstrap (HTML/CSS) code.

## Disclaimer

The following software is an extension of Tony Beltramelli's [pix2code](https://github.com/tonybeltramelli/pix2code),
and is used solely for educational purposes. The [pix2code web dataset](https://github.com/tonybeltramelli/pix2code/tree/master/datasets) is used to train, test, and validate my deep learning model.

This project is a personal research project that demonstrates an application of deep neural networks in generating content (bootstrap code) given pairs of visual and textual data. It has given me a chance to explore the different aspects of deep learning from learning different deep learning architectures to understanding the fundamentals of training a neural network. This whole project has been a great learning experience for me, and has elevated my interests specifically in the domain of deep learning. So I want to acknowledge and thank Tony Beltramelli and his work in pix2code.

## Setup

### Prerequisites
- Python 3
- pip

### Install dependencies

```sh
pip install -r requirements.txt
```

## Project Structure

```
.
├── base               - contains abstract class of model
├── compiler           - contains DSL compiler to bootstrap
│   ├── assets
│   └── classes
├── config             - contains neural network hyperparameters
├── data               - contains dataset and scripts to prepare data
│   ├── all_data
│   ├── eval
│   ├── img
│   │   ├── eval_images
│   │   ├── test_images
│   │   └── train_images
│   ├── test
│   └── train
├── data_loader        - data generator class inherits from Kera's Sequence
├── demo               - files for quick demo of code generation
│   └── data
│       ├── demo_data
│       └── demo_images
├── evaluator          - evaluation of model based on BLEU scores
├── generator          - code generator to generate DSL and HTML code
├── model              - contains implementation of model architecture
├── results            - contains model files & results of model training
├── trainer            - trainer used to train and fit model
└── utils              - helper functions used for callbacks & tokenizer

```

## Usage & Workflow

- Prepare pix2code web dataset by reassembling and unzipping the web dataset

```sh
cd data
cat all_data.zip.* > all_data.zip
unzip all_data.zip
```

- Partition the dataset by splitting the dataset for training, validation, and testing
  - Training set: 1400 image-markup pairs saved to data/train/
  - Validation set: 100 image-markup pairs saved to data/eval/
  - Testing set: 250 image-markup pairs saved to data/test/
- Convert image data to numpy arrays

```sh
python prepare_data.py
```

- Train model given output path argument to save model files
  - training set and validation set paths have been predefined
  - Output directory of the model will include:
    - Model architecture (JSON file)
    - Model weights (hdf5 files)
    - training.log with acc, loss, val_acc, val_loss for each epoch
    - model.png showing a visualization of the Model
    - config.txt which shows the hyperparameters of the neural network (batch size, epoch size, etc.)
    - logs folder with TensorBoard visualizations of training and validation metrics


```sh
cd trainer
# usage: trainer.py  <output path>
python trainer.py ../results/
```

- Evaluate model using BLEU scores given the model path
  - test set is used to evaluate model and test set path has been predefined
  - Model path must contain:
    - Model architecture represented as JSON file
    - Model weights represented as a hdf5 file
  - will print and write BLEU 4-ngram scores
    - bleu_scores.txt file will be generated in model path

```sh
cd evaluator
# usage: evaluate_model.py  <model path>
python evaluate_model.py ../results/
```

- Generate bootstrap (HTML) code given path to images and model path
  - Will first generate DSL code for the images in the path specified
    - saved to a folder called 'generated_dsl' in model path
  - Then compile to bootstrap (HTML) code
    - saved to a folder called 'generated_html' in model path

```sh
cd generator
# usage: generate_code.py <image path> <model path>
python generate_code.py ../data/img/test_images ../results/
```

## Technical Report

Details of implementation and architecture will be located here on my blog soon: [Kevin Chuang's Blog](https://kevinchuangblog.wordpress.com/)


## Acknowledgements
- Tony Beltramelli's pix2code [paper](https://arxiv.org/pdf/1705.07962.pdf) & [code](https://github.com/tonybeltramelli/pix2code)
- Jason Brownlee's [tutorial](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/) on image captioning

## License

See the [LICENSE](https://github.com/k-chuang/code-generation-from-images/blob/master/LICENSE) file for license rights and limitations (MIT).
