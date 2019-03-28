import tensorflow as tf

from conv_gru import ConvGRUCell
from tf_utils import conv2d_act, down_sampling
import config as c


class Encoder(object):
    def __init__(self, batch, seq):
        if c.DTYPE == "single":
            self._dtype = tf.float32
        elif c.DTYPE == "HALF":
            self._dtype = tf.float16

        self._batch = batch
        self._seq = seq
        self._h = c.H
        self._w = c.W
        self._in_c = c.IN_CHANEL
        self.rnn_blocks = []
        self.rnn_states = []

        self.build_rnn_blocks()

    def build_rnn_blocks(self):
        """
        first rnn changes input chanels
        input (b, 180, 180, 8) output (b, 180, 180, 64)
        so set the chanel parameter to define gru i2h.
        other rnn cells keep the input chanel.
        :return:
        """
        for i in range(len(c.NUM_FILTER)):
            if i == 0:
                chanel = c.FIRST_CONV[0]
            else:
                chanel = c.NUM_FILTER[i]
            self.rnn_blocks.append(ConvGRUCell(num_filter=c.NUM_FILTER[i],
                                               b_h_w=(self._batch,
                                                      c.FEATMAP_SIZE[i],
                                                      c.FEATMAP_SIZE[i]),
                                               h2h_kernel=c.H2H_KERNEL[i],
                                               i2h_kernel=c.I2H_KERNEL[i],
                                               name="e_cgru_" + str(i),
                                               chanel=chanel))

    def init_rnn_states(self):
        for block in self.rnn_blocks:
            self.rnn_states.append(block.zero_state())

    def stack_rnn_encoder(self, in_data):
        with tf.variable_scope("Encoder"):

            rnn_block_num = len(c.NUM_FILTER)

            data = tf.reshape(in_data, shape=(self._seq * self._batch,
                                              self._h, self._w, self._in_c))
            conv1 = conv2d_act(data,
                               kernel=c.FIRST_CONV[1],
                               strides=c.FIRST_CONV[2],
                               num_filters=c.FIRST_CONV[0],
                               name="first_conv1")

            for i in range(rnn_block_num):
                if i == 0:
                    c1_s = conv1.shape.as_list()
                    conv1 = tf.reshape(conv1, shape=(self._batch, self._seq, c1_s[-3], c1_s[-2], c1_s[-1]))
                    input = conv1
                else:
                    input = downsample
                print(input)
                output, states = self.rnn_blocks[i].unroll(length=self._seq,
                                                           inputs=input,
                                                           begin_state=None)
                print(output)
                self.rnn_states.append(states)
                if i < rnn_block_num - 1:
                    downsample = down_sampling(output,
                                               kshape=c.DOWNSAMPLE[i][0],
                                               stride=c.DOWNSAMPLE[i][1],
                                               num_filters=c.NUM_FILTER[i + 1],
                                               name="down_sample_" + str(i))
