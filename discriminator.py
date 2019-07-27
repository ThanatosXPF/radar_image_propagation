import tensorflow as tf
from encoder import Encoder
from forecaster import Forecaster
from tensorflow.contrib.layers import xavier_initializer

from config import c


class Discriminator:
    def __init__(self, session, summary):
        if c.DTYPE == "single":
            self._dtype = tf.float32
        elif c.DTYPE == "HALF":
            self._dtype = tf.float16

        self._batch = c.BATCH_SIZE
        self._out_seq = c.OUT_SEQ
        self._in_seq = c.IN_SEQ

        self._h = c.H
        self._w = c.W

        self.pred_data = None
        self.real_data = None

        self.sess = session
        self.summary_writer = summary

        self.define_graph()

    def define_graph(self):
        with tf.name_scope("Discriminator") and tf.device("/device:GPU:1"):
            with tf.name_scope("graph"):
                self.pred_data = tf.placeholder(self._dtype, (self._batch, self._out_seq, self._h, self._w, 1))
                self.real_data = tf.placeholder(self._dtype, (self._batch, self._out_seq, self._h, self._w, 1))

                d_pred = self.encoder_decoder(self.pred_data)
                d_real = self.encoder_decoder(self.real_data, reuse=True)


            with tf.name_scope("Discriminator_loss"):
                self.global_step = tf.Variable(0, trainable=False)

                self.d_pred = d_pred

                mse_real = tf.reduce_mean(tf.square(d_real - self.real_data))
                mse_pred = tf.reduce_mean(tf.square(d_pred - self.pred_data))

                self.mse_real, self.mse_pred = mse_real, mse_pred

                self.loss = mse_real + tf.maximum(1000 - mse_pred, 0)

                self.summary = tf.summary.merge([
                    tf.summary.scalar('d_loss', self.loss),
                    tf.summary.scalar('mse_real', self.mse_real),
                    tf.summary.scalar('mse_pred', self.mse_pred)
                ])

                self.optim = tf.train.AdamOptimizer(c.LR).minimize(self.loss,
                                                                   global_step=self.global_step,
                                                                   name="discriminator_op")
    def encoder_decoder(self, data, reuse=None):
        with tf.variable_scope("Graph", reuse=tf.AUTO_REUSE):
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
                for i in range(self._out_seq):
                    encoder_net.rnn_encoder(data[:, i, ...])
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
        return pred

    def train_step(self, in_data, gt_data, g_model):
        g_feed_dict = {g_model.in_data: in_data, g_model.gt_data: gt_data}
        pred = g_model.sess.run(g_model.result, feed_dict=g_feed_dict)
        real = gt_data

        feed_dict = {self.pred_data: pred,
                     self.real_data: real}

        _, loss, summaries, mse_real, mse_pred, global_step = \
            self.sess.run([self.optim,
                           self.loss,
                           self.summary,
                           self.mse_real,
                           self.mse_pred,
                           self.global_step], feed_dict=feed_dict)

        print("D_model", mse_pred, mse_real, loss)

        if global_step % c.SUMMARY_ITER == 0:
            self.summary_writer.add_summary(summaries, global_step)

        return loss, mse_real, mse_pred
