from base.base_model import BaseModel
import tensorflow as tf
from data_loader.data_generator import data_generator
import tensorflow.contrib.seq2seq as seq2seq

class CodeGeneratorModel(BaseModel):
    '''
    Code Generator Neural Network Model with an encoder layer (CNN + GRU) and a decoder layer
    '''
    def __init__(self, config, is_training):
        super(CodeGeneratorModel, self).__init__(config)
        self.is_training = is_training
        self.max_length = self.config.max_length
        self.embedding_size = self.config.embedding_size
        self.vocab_size = self.config.vocab_size
        self.batch_size = self.config.batch_size
        self.build_graph()

    def build_graph(self):
        """Define computational graph by building model"""
        # init = tf.contrib.layers.xavier_initializer()
        self._init_placeholders()
        encoder_convnet_out = self.encoder_convnet(self.encoder_cnn_inputs, self.is_training, self.max_length)
        encoder_gru_out = self.encoder_gru(self.encoder_sequence_inputs, self.vocab_size)
        self.predictions, self.logits = self.decoder_GRU(encoder_convnet_out, encoder_gru_out, self.vocab_size)
        self._init_accuracy()
        self._init_loss()
        self._init_optimizer()



    def _init_placeholders(self):
        self.encoder_cnn_inputs = tf.placeholder(
            shape=(None, 256, 256, 3),
            dtype=tf.float32,
            name='encoder_cnn_inputs'
        )

        self.encoder_sequence_inputs = tf.placeholder(
            shape=(None, self.max_length),
            dtype=tf.int32,
            name='encoder_sequence_inputs'
        )

        self.decoder_targets = tf.placeholder(
            shape=(None, self.vocab_size),
            dtype=tf.int32,
            name='decoder_targets',
        )

    @staticmethod
    def encoder_gru(input_texts, vocab_size):
        with tf.variable_scope("encoder_gru"):
            with tf.device('/cpu:0'):
                # Create the embedding variable (each row represent a word embedding vector)
                embedding = tf.get_variable("embedding", [vocab_size, 50])
                # embedding = tf.Variable(tf.random_normal([self.vocab_size, 50]))
                # Lookup the corresponding embedding vectors for each sample in X
                X_embed = tf.nn.embedding_lookup(embedding, input_texts)

            # sequence_len = tf.count_nonzero(input_texts, axis=-1)
            gru_layers = [tf.contrib.rnn.GRUCell(num_units=units) for units in [128, 128]]
            # create a RNN cell composed sequentially of a number of RNNCells
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(gru_layers)
            # 'outputs' is a tensor of shape [batch_size, max_time, 256]
            # 'state' is a N-tuple where N is the number of LSTMCells containing a
            # tf.contrib.rnn.LSTMStateTuple for each cell
            outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                               inputs=X_embed)

        return outputs

    @staticmethod
    def encoder_convnet(input_images, is_training, max_length):
        with tf.variable_scope("encoder_convnet"):
            # x = tf.reshape(x_image, shape=[None, 3, 256, 256])
            conv1 = tf.layers.conv2d(input_images, 16, 3, padding='valid', activation='relu')
            conv2 = tf.layers.conv2d(conv1, 16, 3, strides=2, padding='same', activation='relu')
            conv3 = tf.layers.conv2d(conv2, 32, 3, padding='same', activation='relu')
            conv4 = tf.layers.conv2d(conv3, 32, 3, strides=2, padding='same', activation='relu')
            conv5 = tf.layers.conv2d(conv4, 64, 3, padding='same', activation='relu')
            conv6 = tf.layers.conv2d(conv5, 64, 3, strides=2, padding='same', activation='relu')
            conv7 = tf.layers.conv2d(conv6, 128, 3, padding='same', activation='relu')
            flatten = tf.layers.flatten(conv7)
            fc1 = tf.layers.dense(flatten, 1024, activation='relu')
            drop1 = tf.layers.dropout(fc1, rate=0.3, training=is_training)
            fc2 = tf.layers.dense(drop1, 1024, activation='relu')
            drop2 = tf.layers.dropout(fc2, rate=0.3, training=is_training)
            # out = tf.tile(drop2, max_length)
            repeat_vec = tf.reshape(tf.tile(tf.expand_dims(drop2, axis=1), [1, 1, 48]), [1, 48, 1024])
            shape = tf.TensorShape([None]).concatenate(repeat_vec.get_shape()[1:])
            out = tf.placeholder_with_default(repeat_vec, shape=shape)
            # out = tf.expand_dims(out, axis=0)
        return out

    @staticmethod
    def decoder_GRU(encoder_convnet_out, encoder_gru_out, vocab_size):
        with tf.variable_scope("decoder_gru"):
            concat1 = tf.concat([encoder_convnet_out, encoder_gru_out], -1)
            gru_layers = [tf.contrib.rnn.GRUCell(num_units=units) for units in [512, 512]]
            # create a RNN cell composed sequentially of a number of RNNCells
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(gru_layers)
            # 'outputs' is a tensor of shape [batch_size, max_time, 256]
            # 'state' is a N-tuple where N is the number of LSTMCells containing a
            # tf.contrib.rnn.LSTMStateTuple for each cell
            outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                               inputs=concat1)
            logits = tf.layers.dense(state.h, vocab_size)
            out = tf.nn.softmax(logits)
            # out = tf.layers.dense(state.h, self.vocab_size, activation='softmax')

        return out, logits

    def inference(self, x):
        pass

    def _init_accuracy(self):
        acc_op = tf.metrics.accuracy(labels=tf.argmax(self.decoder_targets, axis=1), predictions=self.predictions)
        self.acc_op = acc_op
        # correct_prediction = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.decoder_targets, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _init_loss(self):
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cast(self.decoder_targets, dtype=tf.int32), logits=self.logits)
        self.loss_op = tf.reduce_mean(self.loss)

    def _init_optimizer(self):
        # self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.decoder_targets, logits=self.logits)
        # tf.summary.scalar('loss', self.loss)
        # self.summary_op = tf.summary.merge_all()
        learning_rate = 0.0001
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients, global_step=tf.train.get_global_step())

