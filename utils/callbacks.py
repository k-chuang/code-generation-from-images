import os
from keras.callbacks import ModelCheckpoint, CSVLogger


def generate_callbacks(output_path):
    filepath = os.path.join(output_path,
            'weights.epoch-{epoch:02d}--loss-{loss:.4f}--acc-{acc:.4f}--val_loss-{val_loss:.4f}--val_acc-{val_acc:.4f}.hdf5')
    # checkpoint = ModelCheckpoint(filepath, verbose=1, save_weights_only=True, period=2)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                 verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='min')
    # Create a callback that streams epoch results to a csv file.
    csv_file = os.path.join(output_path, 'training.log')
    # csv_file = 'weights/training.log'
    csv_logger = CSVLogger(csv_file)

    callbacks_list = [checkpoint, csv_logger]
    return callbacks_list