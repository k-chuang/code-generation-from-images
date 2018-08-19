import os
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
import tensorflow as tf


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


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
    log_path = os.path.join(output_path, 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    callbacks_list = [checkpoint, csv_logger, TrainValTensorBoard(log_dir=log_path, write_graph=False)]
    return callbacks_list