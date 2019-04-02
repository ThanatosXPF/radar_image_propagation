import tensorflow as tf

from encoder import Encoder
from forecaster import Forecaster
from evaluation import get_loss_weight_symbol, \
    weighted_mse, weighted_mae, gdl_loss
import config as c


class Model(object):
    def __init__(self, restore_path=None, mode="train"):
        if c.DTYPE == "single":
            self._dtype = tf.float32
        elif c.DTYPE == "HALF":
            self._dtype = tf.float16

        if mode == "train":
            self._batch = c.BATCH_SIZE
        else:
            self._batch = c.BATCH_SIZE

        self._in_seq = c.IN_SEQ
        if mode == "train":
            self._out_seq = c.OUT_SEQ
        else:
            self._out_seq = c.PREDICT_LENGTH
        self._h = c.H
        self._w = c.W
        self._in_c = c.IN_CHANEL

        self.define_graph()

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

        self.saver = tf.train.Saver(max_to_keep=0)

        if restore_path is not None:
            self.saver.restore(self.sess, restore_path)

    def init_params(self):
        self.sess.run(tf.global_variables_initializer())

    def define_graph(self):
        with tf.variable_scope("Graph", reuse=tf.AUTO_REUSE):
            self.in_data = tf.placeholder(self._dtype,
                                          shape=(self._batch, self._in_seq,
                                                 self._h, self._w, self._in_c),
                                          name="input")
            self.gt_data = tf.placeholder(self._dtype,
                                          shape=(self._batch, self._out_seq,
                                                 self._h, self._w, 1),
                                          name="gt")
            self.global_step = tf.Variable(0, trainable=False)
            with tf.device('/device:GPU:0'):
                encoder_net = Encoder(self._batch,
                                      self._in_seq,
                                      gru_fms=c.GRU_FMS,
                                      gru_filter=c.ENCODER_GRU_FILTER,
                                      gru_in_chanel=c.ENCODER_GRU_INCHANEL,
                                      conv_fms=c.CONV_FMS,
                                      conv_stride=c.CONV_STRIDE,
                                      h2h_kernel=c.H2H_KERNEL,
                                      i2h_kernel=c.I2H_KERNEL)
                for i in range(self._in_seq):
                    encoder_net.rnn_encoder(self.in_data[:, i, ...])
                states = encoder_net.rnn_states
            with tf.device('/device:GPU:1'):
                forecaster_net = Forecaster(self._batch,
                                            self._out_seq,
                                            gru_fms=c.GRU_FMS,
                                            gru_filter=c.DECODER_GRU_FILTER,
                                            gru_in_chanel=c.DECODER_GRU_INCHANEL,
                                            conv_fms=c.DECONV_FMS,
                                            conv_stride=c.DECONV_STRIDE,
                                            infer_shape=c.DECODER_INFER_SHAPE,
                                            h2h_kernel=c.H2H_KERNEL,
                                            i2h_kernel=c.I2H_KERNEL,
                                            rnn_states=states)

                for i in range(self._out_seq):
                    forecaster_net.rnn_forecaster()
                pred = tf.concat(forecaster_net.pred, axis=1)

            with tf.variable_scope("loss"):
                gt = self.gt_data
                weights = get_loss_weight_symbol(pred)
                self.result = pred
                self.mse = weighted_mse(pred, gt, weights)
                self.mae = weighted_mae(pred, gt, weights)
                self.gdl = gdl_loss(pred, gt)
                loss = c.L1_LAMBDA * self.mse + c.L2_LAMBDA * self.mae + c.GDL_LAMBDA * self.gdl
                self.optimizer = tf.train.AdamOptimizer(c.LR).minimize(loss, global_step=self.global_step)

    def train_step(self, in_data, gt_data):
        _, mse, mae, gdl, pred = self.sess.run([self.optimizer, self.mse, self.mae, self.gdl, self.result],
                                               feed_dict={
                                                   self.in_data: in_data,
                                                   self.gt_data: gt_data
                                               })
        print("pred: ", pred.min(), pred.max())
        print("gt: ", gt_data.min(), gt_data.max())
        return mse, mae, gdl

    def valid_step(self, in_data, gt_data):
        mse, mae, gdl, pred = self.sess.run([self.mse, self.mae, self.gdl, self.result],
                                            feed_dict={
                                                self.in_data: in_data,
                                                self.gt_data: gt_data
                                            })
        print("pred: ", pred.min(), pred.max())
        print("gt: ", gt_data.min(), gt_data.max())
        return mse, mae, gdl, pred

    def save_model(self):
        from os.path import join
        save_path = self.saver.save(self.sess,
                                    join(c.SAVE_MODEL, "model.ckpt"),
                                    global_step=self.global_step)
        print("Model saved in path: %s" % save_path)
