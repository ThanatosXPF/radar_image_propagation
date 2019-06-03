import numpy as np
import tensorflow as tf

from encoder import Encoder
from forecaster import Forecaster
from evaluation import get_loss_weight_symbol, \
    weighted_mse, weighted_mae, gdl_loss
from config import c


class Generator(object):
    def __init__(self, session, summary, mode="train"):
        if c.DTYPE == "single":
            self._dtype = tf.float32
        elif c.DTYPE == "HALF":
            self._dtype = tf.float16

        self._batch = c.BATCH_SIZE

        if mode == "train":
            self._out_seq = c.OUT_SEQ
            self._h = c.H
            self._w = c.W
        else:
            # self._batch = 1
            self._out_seq = c.PREDICT_LENGTH
            self._h = c.PREDICTION_H
            self._w = c.PREDICTION_W

        self._in_seq = c.IN_SEQ
        # self._h = c.H
        # self._w = c.W
        self._in_c = c.IN_CHANEL
        self.adv_involve = tf.constant(c.ADV_INVOLVE)
        self.gl_st = 0

        self.define_graph()

        self.sess = session
        self.summary_writer = summary

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
                                      gru_filter=c.ENCODER_GRU_FILTER,
                                      gru_in_chanel=c.ENCODER_GRU_INCHANEL,
                                      conv_kernel=c.CONV_KERNEL,
                                      conv_stride=c.CONV_STRIDE,
                                      h2h_kernel=c.H2H_KERNEL,
                                      i2h_kernel=c.I2H_KERNEL,
                                      height=self._h,
                                      width=self._w)
                for i in range(self._in_seq):
                    encoder_net.rnn_encoder(self.in_data[:, i, ...])
            states = encoder_net.rnn_states
            with tf.device('/device:GPU:1'):
                forecaster_net = Forecaster(self._batch,
                                            self._out_seq,
                                            gru_filter=c.DECODER_GRU_FILTER,
                                            gru_in_chanel=c.DECODER_GRU_INCHANEL,
                                            deconv_kernel=c.DECONV_KERNEL,
                                            deconv_stride=c.DECONV_STRIDE,
                                            h2h_kernel=c.H2H_KERNEL,
                                            i2h_kernel=c.I2H_KERNEL,
                                            rnn_states=states,
                                            height=self._h,
                                            width=self._w)

                for i in range(self._out_seq):
                    forecaster_net.rnn_forecaster()
            pred = tf.concat(forecaster_net.pred, axis=1)

            with tf.variable_scope("loss"):
                gt = self.gt_data
                weights = get_loss_weight_symbol(pred)

                self.result = pred
                self.mse = tf.reduce_mean(tf.square(pred - gt))
                self.mae = weighted_mae(pred, gt, weights)
                self.gdl = gdl_loss(pred, gt)
                self.d_loss = self.result

                if c.ADVERSARIAL:
                    self.d_pred = tf.placeholder(self._dtype, (self._batch, self._out_seq, self._h, self._w, 1))
                    self.d_loss = tf.reduce_mean(tf.square(self.d_pred - gt))
                    self.loss = tf.cond(self.global_step > self.adv_involve,
                                        lambda: c.L1_LAMBDA * self.mae \
                                                + c.L2_LAMBDA * self.mse \
                                                + c.GDL_LAMBDA * self.gdl \
                                                + c.ADV_LAMBDA * self.d_loss,
                                        lambda: c.L1_LAMBDA * self.mae \
                                                + c.L2_LAMBDA * self.mse \
                                                + c.GDL_LAMBDA * self.gdl
                                        )
                    # self.loss = c.L1_LAMBDA * self.mae \
                    #        + c.L2_LAMBDA * self.mse \
                    #        + c.GDL_LAMBDA * self.gdl \
                    #        + c.ADV_LAMBDA * self.d_loss
                else:
                    self.loss = c.L1_LAMBDA * self.mae + c.L2_LAMBDA * self.mse + c.GDL_LAMBDA * self.gdl

                self.optimizer = tf.train.AdamOptimizer(c.LR).minimize(self.loss, global_step=self.global_step)

                self.summary = tf.summary.merge([tf.summary.scalar('mse', self.mse),
                                                 tf.summary.scalar('mae', self.mae),
                                                 tf.summary.scalar('gdl', self.gdl),
                                                 tf.summary.scalar('combine_loss', self.loss)])

    def train_step(self, in_data, gt_data, d_model=None):
        feed_dict = {self.in_data: in_data, self.gt_data: gt_data}
        if c.ADVERSARIAL:
            if self.gl_st > c.ADV_INVOLVE:
                g_pred = self.sess.run(self.result, feed_dict=feed_dict)

                d_feed_dict = {d_model.real_data: gt_data, d_model.pred_data: g_pred}

                d_pred = d_model.sess.run(d_model.d_pred, feed_dict=d_feed_dict)

                feed_dict[self.d_pred] = d_pred
            else:
                feed_dict[self.d_pred] = np.zeros((self._batch, self._out_seq, self._h, self._w, 1))
        _, loss, mse, d_loss, pred, summary, global_step = \
            self.sess.run([self.optimizer,
                           self.loss,
                           self.mse,
                           self.d_loss,
                           self.result,
                           self.summary,
                           self.global_step],
                          feed_dict=feed_dict)

        self.gl_st = global_step
        print("pred: ", pred.min(), pred.max())
        print("gt: ", gt_data.min(), gt_data.max())
        if global_step % c.SUMMARY_ITER == 0:
            self.summary_writer.add_summary(summary, global_step)
            print("Save summaries")
        return loss, mse, d_loss, global_step

    def valid_step(self, in_data, gt_data):
        mse, mae, gdl, pred = self.sess.run([self.mse,
                                             self.mae,
                                             self.gdl,
                                             self.result],
                                            feed_dict={
                                                self.in_data: in_data,
                                                self.gt_data: gt_data
                                            })
        print("pred: ", pred.min(), pred.max())
        print("gt: ", gt_data.min(), gt_data.max())
        return mse, mae, gdl, pred
