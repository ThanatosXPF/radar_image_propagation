import tensorflow as tf

from conv_gru import ConvGRUCell
import config as c

from tf_utils import up_sampling, deconv2d_act, conv2d, conv2d_act


class Forecaster(object):
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
        # self.rnn_states = states

        self.build_rnn_blocks()

    def build_rnn_blocks(self):
        """
        same as encoder
        first rnn input (b, 180, 180 ,192) output (b, 180, 180, 64)
        :return:
        """
        for i in range(len(c.NUM_FILTER)):
            if i == 0:
                chanel = c.NUM_FILTER[1]
            else:
                chanel = c.NUM_FILTER[i]
            print(i)
            self.rnn_blocks.append(ConvGRUCell(num_filter=c.NUM_FILTER[i],
                                               b_h_w=(self._batch,
                                                      c.FEATMAP_SIZE[i],
                                                      c.FEATMAP_SIZE[i]),
                                               h2h_kernel=c.H2H_KERNEL[i],
                                               i2h_kernel=c.I2H_KERNEL[i],
                                               name="f_cgru_" + str(i),
                                               chanel=chanel))

    def stack_rnn_forecaster(self, block_state_list):
        with tf.variable_scope("Forecaster"):
            rnn_block_num = len(c.NUM_FILTER)
            rnn_block_output = []
            curr_inputs = None
            for i in range(rnn_block_num - 1, -1, -1):
                rnn_out, rnn_state = self.rnn_blocks[i].unroll(length=self._seq,
                                                               inputs=curr_inputs,
                                                               begin_state=block_state_list[i],
                                                               )
                rnn_block_output.append(rnn_out)

                if i > 0:
                    print(rnn_out)
                    upsample = up_sampling(rnn_out,
                                           kshape=c.UPSAMPLE[i - 1][0],
                                           stride=c.UPSAMPLE[i - 1][1],
                                           num_filter=c.NUM_FILTER[i],
                                           name="Up_sample_" + str(i))
                    curr_inputs = upsample

            deconv1 = deconv2d_act(rnn_block_output[-1],
                                   kernel=c.LAST_DECONV[1],
                                   stride=c.LAST_DECONV[2],
                                   num_filters=c.LAST_DECONV[0],
                                   use_bias=False,
                                   dtype=self._dtype,
                                   name="last_conv")

            conv_final = conv2d_act(deconv1, kernel=3, strides=1, num_filters=8, name="conv_final")
            pred = conv2d(conv_final, kshape=(1, 1, 8, 1), name="out")
            pred = tf.reshape(pred, shape=(self._batch, self._seq, self._h, self._w, self._in_c))
            self.pred = pred
