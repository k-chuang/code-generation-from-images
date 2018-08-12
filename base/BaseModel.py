import os
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
from contextlib import redirect_stdout
import glob

class BaseModel:
    def __init__(self, input_shape, output_path):
        self.model = None
        self.input_shape = input_shape
        self.output_path = output_path
        self.name = ""

    def save_model(self):
        model_json = self.model.to_json(indent=4)
        with open("{}/{}.json".format(self.output_path, self.name), "w") as json_file:
            json_file.write(model_json)

    def save_model_weights(self):
        self.model.save_weights("{}/{}.hdf5".format(self.output_path, self.name))

    def load_model(self):
        # output_name = self.name if name == "" else name
        model_json = glob.glob(os.path.join(self.output_path, '*.json'))[0]
        # with open("{}/{}.json".format(self.output_path, model_json), "r") as json_file:
        with open(model_json, "r") as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)

    def load_model_weights(self):
        # output_name = self.name if name == "" else name
        model_weights_path = glob.glob(os.path.join(self.output_path, '*.hdf5'))[0]
        self.model.load_weights(model_weights_path)
        # self.model.load_weights("{}/{}.hdf5".format(self.output_path, model_weights_path))

    def summarize(self):
        with open(os.path.join(self.output_path, "summary.txt"), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()
            # self.model.summary(print_fn=lambda x: f.write(x + '\n'))

    def plot_model(self):
        plot_model(self.model, to_file=os.path.join(self.output_path, 'model.png'), show_shapes=True)
