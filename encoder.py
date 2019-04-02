import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from conv_gru import ConvGRUCell
from tf_utils import conv2d_act
import config as c


class Encoder(object):
    def __init__(self, batch, seq, gru_fms, gru_filter, gru_in_chanel, conv_fms, conv_stride, h2h_kernel, i2h_kernel):
        if c.DTYPE == "single":
            self._dtype = tf.float32
        elif c.DTYPE == "HALF":
            self._dtype = tf.float16

        self._batch = batch
        self._seq = seq
        self._h = c.H
        self._w = c.W
        self._in_c = c.IN_CHANEL

        self.stack_num = len(gru_filter)
        self.rnn_blocks = []
        self.rnn_states = []
        self.conv_kernels = []
        self.conv_bias = []
        self.conv_stride = conv_stride

        self._gru_fms = gru_fms
        self._gru_filter = gru_filter
        self._conv_fms = conv_fms
        self._gru_in_chanel = gru_in_chanel
        self._h2h_kernel = h2h_kernel
        self._i2h_kernel = i2h_kernel

        self.build_rnn_blocks()
        self.init_rnn_states()
        self.init_parameters()

    def build_rnn_blocks(self):
        """
        first rnn changes input chanels
        input (b, 180, 180, 8) output (b, 180, 180, 64)
        so set the chanel parameter to define gru i2h.
        other rnn cells keep the input chanel.
        :return:
        """
        with tf.variable_scope("Encoder"):
            for i in range(len(self._gru_fms)):
                self.rnn_blocks.append(ConvGRUCell(num_filter=self._gru_filter[i],
                                                   b_h_w=(self._batch,
                                                          self._gru_fms[i],
                                                          self._gru_fms[i]),
                                                   h2h_kernel=self._h2h_kernel[i],
                                                   i2h_kernel=self._i2h_kernel[i],
                                                   name="e_cgru_" + str(i),
                                                   chanel=self._gru_in_chanel[i]))

    def init_parameters(self):
        with tf.variable_scope("Encoder"):
            for i in range(len(self._conv_fms)):
                self.conv_kernels.append(tf.get_variable(name=f"Conv{i}_W",
                                                         shape=self._conv_fms[i],
                                                         initializer=xavier_initializer(uniform=False),
                                                         dtype=self._dtype))
                self.conv_bias.append(tf.get_variable(name=f"Conv{i}_b",
                                                      shape=self._conv_fms[-1]))

    def init_rnn_states(self):
        for block in self.rnn_blocks:
            self.rnn_states.append(block.zero_state())

    def rnn_encoder(self, in_data):
        with tf.variable_scope("Encoder"):
            for i in range(self.stack_num):
                conv = conv2d_act(input=in_data,
                                  name=f"Conv{i}",
                                  kernel=self.conv_kernels[i],
                                  bias=self.conv_bias[i],
                                  strides=self.conv_stride[i])

                output, states = self.rnn_blocks[i](inputs=conv,
                                                    states=self.rnn_states[i])
                self.rnn_states[i] = states
                in_data = output
