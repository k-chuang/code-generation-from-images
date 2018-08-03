from keras.models import Sequential, Model
from keras.layers import Embedding, TimeDistributed, \
    RepeatVector, LSTM, concatenate, Input, Reshape, Dense, GRU, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import RMSprop, Adam
from base.base_model import BaseModel
from config.config import *


class CodeGeneratorModel(BaseModel):
    def __init__(self, input_shape, output_path):
        super(CodeGeneratorModel, self).__init__(input_shape, output_path)
        self.name = "CodeGeneratorModel"

        # Create the encoder
        image_model = Sequential()
        image_model.add(Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
        image_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        image_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        image_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        image_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        image_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        image_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

        image_model.add(Flatten())
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))

        image_model.add(RepeatVector(CONTEXT_LENGTH))

        visual_input = Input(shape=(256, 256, 3,))
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
        self.model = Model(inputs=[visual_input, language_input], outputs=decoder)
        # optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        optimizer = Adam(lr=0.0001, clipvalue=1.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def fit(self, images, partial_captions, next_words, callbacks=None, **kwargs):
        self.model.fit([images, partial_captions], next_words, shuffle=False, epochs=EPOCHS,
                       callbacks=callbacks, batch_size=BATCH_SIZE, verbose=1, **kwargs)
        if callbacks is None:
            self.save()

    def fit_generator(self, generator, steps_per_epoch, callbacks=None, **kwargs):
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                                 epochs=EPOCHS, verbose=1, **kwargs)
        if callbacks is None:
            self.save()

    def predict(self, image, partial_caption, **kwargs):
        return self.model.predict([image, partial_caption], verbose=0, **kwargs)[0]

    def predict_batch(self, images, partial_captions, **kwargs):
        return self.model.predict([images, partial_captions], verbose=1, **kwargs)
