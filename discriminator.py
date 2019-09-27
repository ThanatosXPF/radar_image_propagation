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
        self._in_seq = c.IN_SEQ

        self._h = c.PREDICTION_H
        self._w = c.PREDICTION_W
        self.strides = c.CONV_STRIDE

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
                if c.PREDICTION_H != c.H or c.PREDICTION_W != c.W:
                    g_h = (c.H - c.PREDICTION_H) // 2
                    g_w = (c.W - c.PREDICTION_W) // 2
                    p = [[0, 0], [0, 0], [g_h, g_h], [g_w, g_w], [0, 0]]
                    pred_data = tf.pad(self.pred_data, p, "CONSTANT")
                    real_data = tf.pad(self.real_data, p, "CONSTANT")
                else:
                    pred_data = self.pred_data
                    real_data = self.real_data

                pred = tf.transpose(pred_data, [0, 4, 2, 3, 1])
                real = tf.transpose(real_data, [0, 4, 2, 3, 1])

                # data = tf.concat([pred, real], axis=0)
                pred_data = tf.reshape(pred, (self._batch, c.H, c.W, self._out_seq))
                d_pred = self.encoder_decoder(pred_data)
                real_data = tf.reshape(real, (self._batch, c.H, c.W, self._out_seq))
                d_real = self.encoder_decoder(real_data, reuse=True)

            with tf.name_scope("Discriminator_loss"):
                self.global_step = tf.Variable(0, trainable=False)

                mid = tf.transpose(d_pred, [0, 3, 1, 2])
                self.d_pred = tf.reshape(mid, (self._batch, self._out_seq, c.H, c.W, 1))

                if c.PREDICTION_H != c.H or c.PREDICTION_W != c.W:
                    self.d_pred = self.d_pred[:, :, 2:-2, 2:-2, :]

                mse_real = tf.reduce_mean(tf.square(d_real - real))
                mse_pred = tf.reduce_mean(tf.square(d_pred - pred))

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

    def auto_encoder_decoder(self, data, reuse=None):
        strides = self.strides
        if len(strides) == 3:
            fms = [7, 5, 3]
            c_channel = [[10, 16], [16, 32], [32, 32]]
            d_channel = [[32, 32], [16, 16], [10, 10]]
        else:
            fms = [7, 5, 5, 3, 3]
            c_channel = [[10, 16], [16, 16], [16, 32], [32, 32], [32, 64]]
            d_channel = [[32, 32], [32, 32], [16, 16], [16, 16], [10, 10]]

        for i, (s, k, c) in enumerate(zip(strides, fms, c_channel)):
            conv1 = tf.layers.conv2d(data, c[0], 3, (1, 1),
                                     activation=tf.nn.leaky_relu,
                                     padding="SAME",
                                     kernel_initializer=xavier_initializer(uniform=False),
                                     name=f"conv_{i}_1",
                                     reuse=reuse)

            bn1 = tf.layers.batch_normalization(conv1, name=f"bn_{i}_1", reuse=reuse)

            conv2 = tf.layers.conv2d(bn1, c[1], k, (s, s),
                                     activation=tf.nn.leaky_relu,
                                     padding="SAME",
                                     kernel_initializer=xavier_initializer(uniform=False),
                                     name=f"conv_{i}_2",
                                     reuse=reuse)

            data = tf.layers.batch_normalization(conv2, name=f"bn_{i}_2", reuse=reuse)

        for i, (s, k, c) in enumerate(zip(strides[::-1], fms[::-1], d_channel)):
            deconv1 = tf.layers.conv2d_transpose(data, c[0], k, (s, s),
                                                 activation=tf.nn.leaky_relu,
                                                 padding="SAME",
                                                 kernel_initializer=xavier_initializer(uniform=False),
                                                 name=f"deconv_{i}_1",
                                                 reuse=reuse)

            dec_bn1 = tf.layers.batch_normalization(deconv1, name=f"dec_bn_{i}_1", reuse=reuse)

            deconv2 = tf.layers.conv2d_transpose(dec_bn1, c[1], 3, (1, 1),
                                                 activation=tf.nn.leaky_relu,
                                                 padding="SAME",
                                                 kernel_initializer=xavier_initializer(uniform=False),
                                                 name=f"deconv_{i}_2",
                                                 reuse=reuse)
            if i != len(strides) - 1 :
                data = tf.layers.batch_normalization(deconv2, name=f"dec_bn_{i}_2", reuse=reuse)
            else:
                data = deconv2
        return data

    def encoder_decoder(self, data, reuse=None):
        conv1 = tf.layers.conv2d(data, 10, 3, (1, 1),
                                 activation=tf.nn.leaky_relu,
                                 padding="SAME",
                                 kernel_initializer=xavier_initializer(uniform=False),
                                 name="conv1",
                                 reuse=reuse)

        bn1 = tf.layers.batch_normalization(conv1, name="bn1", reuse=reuse)

        conv2 = tf.layers.conv2d(bn1, 16, 7, (5, 5),
                                 activation=tf.nn.leaky_relu,
                                 padding="SAME",
                                 kernel_initializer=xavier_initializer(uniform=False),
                                 name="conv2",
                                 reuse=reuse)

        bn2 = tf.layers.batch_normalization(conv2, name="bn2", reuse=reuse)

        conv3 = tf.layers.conv2d(bn2, 16, 3, (1, 1),
                                 activation=tf.nn.leaky_relu,
                                 padding="SAME",
                                 kernel_initializer=xavier_initializer(uniform=False),
                                 name="conv3",
                                 reuse=reuse)

        bn3 = tf.layers.batch_normalization(conv3, name="bn3", reuse=reuse)

        conv4 = tf.layers.conv2d(bn3, 32, 5, (3, 3),
                                 activation=tf.nn.leaky_relu,
                                 padding="SAME",
                                 kernel_initializer=xavier_initializer(uniform=False),
                                 name="conv4",
                                 reuse=reuse)

        bn4 = tf.layers.batch_normalization(conv4, name="bn4", reuse=reuse)

        conv5 = tf.layers.conv2d(bn4, 32, 3, (1, 1),
                                 activation=tf.nn.leaky_relu,
                                 padding="SAME",
                                 kernel_initializer=xavier_initializer(uniform=False),
                                 name="conv5",
                                 reuse=reuse)

        bn5 = tf.layers.batch_normalization(conv5, name="bn5", reuse=reuse)

        conv6 = tf.layers.conv2d(bn5, 32, 3, (2, 2),
                                 activation=tf.nn.leaky_relu,
                                 padding="SAME",
                                 kernel_initializer=xavier_initializer(uniform=False),
                                 name="conv6",
                                 reuse=reuse)

        bn6 = tf.layers.batch_normalization(conv6, name="bn6", reuse=reuse)

        deconv1 = tf.layers.conv2d_transpose(bn6, 32, 3, (2, 2),
                                             activation=tf.nn.leaky_relu,
                                             padding="SAME",
                                             kernel_initializer=xavier_initializer(uniform=False),
                                             name="deconv1",
                                             reuse=reuse)

        dec_bn1 = tf.layers.batch_normalization(deconv1, name="dec_bn1", reuse=reuse)

        deconv2 = tf.layers.conv2d_transpose(dec_bn1, 32, 3, (1, 1),
                                             activation=tf.nn.leaky_relu,
                                             padding="SAME",
                                             kernel_initializer=xavier_initializer(uniform=False),
                                             name="deconv2",
                                             reuse=reuse)

        dec_bn2 = tf.layers.batch_normalization(deconv2, name="dec_bn2", reuse=reuse)

        deconv3 = tf.layers.conv2d_transpose(dec_bn2, 16, 5, (3, 3),
                                             activation=tf.nn.leaky_relu,
                                             padding="SAME",
                                             kernel_initializer=xavier_initializer(uniform=False),
                                             name="deconv3",
                                             reuse=reuse)

        dec_bn3 = tf.layers.batch_normalization(deconv3, name="dec_bn3", reuse=reuse)

        deconv4 = tf.layers.conv2d_transpose(dec_bn3, 16, 3, (1, 1),
                                             activation=tf.nn.leaky_relu,
                                             padding="SAME",
                                             kernel_initializer=xavier_initializer(uniform=False),
                                             name="deconv4",
                                             reuse=reuse)

        dec_bn4 = tf.layers.batch_normalization(deconv4, name="dec_bn4", reuse=reuse)

        deconv5 = tf.layers.conv2d_transpose(dec_bn4, 10, 7, (5, 5),
                                             activation=tf.nn.leaky_relu,
                                             padding="SAME",
                                             kernel_initializer=xavier_initializer(uniform=False),
                                             name="deconv5",
                                             reuse=reuse)

        dec_bn5 = tf.layers.batch_normalization(deconv5, name="dec_bn5", reuse=reuse)

        deconv6 = tf.layers.conv2d_transpose(dec_bn5, self._out_seq, 3, (1, 1),
                                             activation=tf.nn.leaky_relu,
                                             padding="SAME",
                                             kernel_initializer=xavier_initializer(uniform=False),
                                             name="deconv6",
                                             reuse=reuse)
        return deconv6

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


if __name__ == '__main__':
    sess = tf.Session()

    d_model = Discriminator(sess, None)

