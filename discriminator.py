import tensorflow as tf
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
        self._h = c.H
        self._w = c.W

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

                pred = tf.transpose(self.pred_data, [0, 4, 2, 3, 1])
                real = tf.transpose(self.real_data, [0, 4, 2, 3, 1])

                data = tf.concat([pred, real], axis=0)

                data = tf.reshape(data, (self._batch * 2, self._h, self._w, self._out_seq))

                conv1 = tf.layers.conv2d(data, 32, 3, (1, 1),
                                         activation=tf.nn.leaky_relu,
                                         padding="SAME",
                                         kernel_initializer=xavier_initializer(uniform=False))

                bn1 = tf.layers.batch_normalization(conv1)

                conv2 = tf.layers.conv2d(bn1, 32, 3, (2, 2),
                                         activation=tf.nn.leaky_relu,
                                         padding="SAME",
                                         kernel_initializer=xavier_initializer(uniform=False))

                bn2 = tf.layers.batch_normalization(conv2)

                conv3 = tf.layers.conv2d(bn2, 64, 3, (2, 2),
                                         activation=tf.nn.leaky_relu,
                                         padding="SAME",
                                         kernel_initializer=xavier_initializer(uniform=False))

                bn3 = tf.layers.batch_normalization(conv3)

                conv4 = tf.layers.conv2d(bn3, 96, 3, (2, 2),
                                         activation=tf.nn.leaky_relu,
                                         padding="SAME",
                                         kernel_initializer=xavier_initializer(uniform=False))

                bn4 = tf.layers.batch_normalization(conv4)

                conv5 = tf.layers.conv2d(bn4, 128, 3, (2, 2),
                                         activation=tf.nn.leaky_relu,
                                         padding="SAME",
                                         kernel_initializer=xavier_initializer(uniform=False))

                bn5 = tf.layers.batch_normalization(conv5)

                deconv1 = tf.layers.conv2d_transpose(bn5, 128, 3, (2, 2),
                                                     activation=tf.nn.leaky_relu,
                                                     padding="SAME",
                                                     kernel_initializer=xavier_initializer(uniform=False))

                bn6 = tf.layers.batch_normalization(deconv1)

                deconv2 = tf.layers.conv2d_transpose(bn6, 96, 3, (2, 2),
                                                     activation=tf.nn.leaky_relu,
                                                     padding="SAME",
                                                     kernel_initializer=xavier_initializer(uniform=False))

                bn7 = tf.layers.batch_normalization(deconv2)

                deconv3 = tf.layers.conv2d_transpose(bn7, 64, 3, (2, 2),
                                                     activation=tf.nn.leaky_relu,
                                                     padding="SAME",
                                                     kernel_initializer=xavier_initializer(uniform=False))

                bn8 = tf.layers.batch_normalization(deconv3)

                deconv4 = tf.layers.conv2d_transpose(bn8, 32, 3, (2, 2),
                                                     activation=tf.nn.leaky_relu,
                                                     padding="SAME",
                                                     kernel_initializer=xavier_initializer(uniform=False))

                bn9 = tf.layers.batch_normalization(deconv4)

                deconv5 = tf.layers.conv2d_transpose(bn9, self._out_seq, 3, (1, 1),
                                                     activation=tf.nn.leaky_relu,
                                                     padding="SAME",
                                                     kernel_initializer=xavier_initializer(uniform=False))

            with tf.name_scope("Discriminator_loss"):
                self.global_step = tf.Variable(0, trainable=False)

                d_pred, d_real = tf.split(deconv5, 2, axis=0)

                mid = tf.transpose(d_pred, [0, 3, 1, 2])
                self.d_pred = tf.reshape(mid, (self._batch, self._out_seq, self._h, self._w, 1))

                mse_real = tf.reduce_mean(tf.square(d_real - real))
                mse_pred = tf.reduce_mean(tf.square(d_pred - pred))

                self.mse_real, self.mse_pred = mse_real, mse_pred

                self.loss = mse_real + tf.maximum(1 - mse_pred, 0)

                self.summary = tf.summary.merge([
                    tf.summary.scalar('d_loss', self.loss),
                    tf.summary.scalar('mse_real', self.mse_real),
                    tf.summary.scalar('mse_pred', self.mse_pred)
                ])

                self.optim = tf.train.AdamOptimizer(c.LR).minimize(self.loss,
                                                                   global_step=self.global_step,
                                                                   name="discriminator_op")

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
