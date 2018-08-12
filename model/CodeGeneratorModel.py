from keras.models import Sequential, Model
from keras.layers import Embedding, RepeatVector, LSTM, \
    concatenate, Input, Reshape, Dense, GRU, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import Adam
from base.BaseModel import BaseModel
from config.config import *
import os
from contextlib import redirect_stdout


class CodeGeneratorModel(BaseModel):
    def __init__(self, input_shape, output_path, image_file_format='channels_last', kernel_initializer='glorot_uniform'):
        super(CodeGeneratorModel, self).__init__(input_shape, output_path)
        self.name = "CodeGeneratorModel"
        self.image_file_format = image_file_format
        self.kernel_init = kernel_initializer
        # Create the encoder
        image_model = Sequential()

        image_model.add(Conv2D(8, (3, 3), padding='valid', activation='relu', input_shape=input_shape,
                               data_format=self.image_file_format, kernel_initializer=self.kernel_init))
        image_model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2,
                               data_format=self.image_file_format, kernel_initializer=self.kernel_init))
        image_model.add(Conv2D(16, (3, 3), activation='relu', padding='same',
                               data_format=self.image_file_format, kernel_initializer=self.kernel_init))
        image_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2,
                               data_format=self.image_file_format, kernel_initializer=self.kernel_init))
        image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                               data_format=self.image_file_format, kernel_initializer=self.kernel_init))
        image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2,
                               data_format=self.image_file_format, kernel_initializer=self.kernel_init))
        image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                               data_format=self.image_file_format, kernel_initializer=self.kernel_init))

        image_model.add(Flatten(data_format=self.image_file_format))
        image_model.add(Dense(1024, activation='relu', kernel_initializer=self.kernel_init))
        image_model.add(Dropout(0.2))
        image_model.add(Dense(1024, activation='relu', kernel_initializer=self.kernel_init))
        image_model.add(Dropout(0.2))

        image_model.add(RepeatVector(CONTEXT_LENGTH))

        visual_input = Input(shape=input_shape)
        encoded_image = image_model(visual_input)

        language_input = Input(shape=(CONTEXT_LENGTH,))
        language_model = Embedding(VOCAB_SIZE, 50, input_length=CONTEXT_LENGTH, mask_zero=True)(language_input)
        language_model = GRU(128, return_sequences=True)(language_model)
        language_model = GRU(128, return_sequences=True)(language_model)

        # Create the decoder
        decoder = concatenate([encoded_image, language_model])
        decoder = GRU(512, return_sequences=True)(decoder)
        decoder = GRU(512, return_sequences=False)(decoder)
        decoder = Dense(VOCAB_SIZE, activation='softmax')(decoder)

        # Compile the model
        self.image_model = image_model
        self.language_model = language_model
        self.decoder = decoder
        self.model = Model(inputs=[visual_input, language_input], outputs=decoder)
        # Want to clip gradients to prevent exploding gradients
        # since output is RNN with tanh activation, we will clip from -1 to 1
        optimizer = Adam(lr=0.0001, clipvalue=1.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def fit(self, images, partial_captions, next_words, callbacks=None, **kwargs):
        self.model.fit([images, partial_captions], next_words, shuffle=False, epochs=EPOCHS,
                       callbacks=callbacks, batch_size=BATCH_SIZE, verbose=1, **kwargs)
        if callbacks is None:
            self.save_model_weights()

    def fit_generator(self, generator, steps_per_epoch, callbacks=None, **kwargs):
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                                 epochs=EPOCHS, verbose=1, **kwargs)
        if callbacks is None:
            self.save_model_weights()

    def predict(self, image, partial_caption, **kwargs):
        return self.model.predict([image, partial_caption], verbose=1, **kwargs)[0]

    def predict_batch(self, images, partial_captions, **kwargs):
        return self.model.predict([images, partial_captions], verbose=1, **kwargs)

    def summarize_image_model(self):
        with open(os.path.join(self.output_path, "image_model_summary.txt"), 'w') as f:
            with redirect_stdout(f):
                self.image_model.summary()
            # self.image_model.summary(print_fn=lambda x: f.write(x + '\n'))
