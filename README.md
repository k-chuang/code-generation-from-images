# Code Generation from Images

An end-to-end deep neural network designed in Keras (with tensorflow backend) that will transform a screenshot into Bootstrap (HTML/CSS) code.

## Disclaimer

The following software is an extension of Tony Beltramelli's [pix2code](https://github.com/tonybeltramelli/pix2code) and Emil Wallner's [Screenshot-to-code-in-Keras](https://github.com/emilwallner/Screenshot-to-code-in-Keras),
and is used solely for educational purposes. I have altered certain parts of the software, but ultimately, this project is largely influenced by these two open source projects. The pix2code dataset is used to train, test, and validate my deep learning model.

This project is a personal research project that demonstrates an application of deep neural networks in generating content given pairs of visual and textual data. It has given me a chance to explore the different aspects of deep learning from learning different deep learning architectures to understanding the fundamentals of training a neural network. This whole project has been a great learning experience for me, and has elevated my interests specifically in the domain of deep learning.

## Setup

### Prerequisites
- Python 3
- pip
- virtualenv

### Virtual environment setup & install dependencies

```sh
virtualenv venv
source venv/bin/activate
(venv) pip install -r requirements.txt
```

## Usage

- Prepare pix2code web dataset by reassembling and unzipping the web dataset

```sh
cd data
zip -F all_data.zip --out all_data.zip
unzip all_data.zip
```





## Project Structure

To be added...

## Acknowledgements
- Tony Beltramelli's pix2code
- Emil Wallner's Screenshot to code in Keras
- Jason Brownlee's tutorials on image captioning
