import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from config import c


class ConvSTLSTMCell(object):
    def __init__(self, num_filter, b_h_w,
                 kernel, name, chanel, dtype=tf.float32):
        self._name = name
        self._batch, self._h, self._w = b_h_w
        self._num_filter = num_filter
        self._dtype = dtype
        self._k_h = kernel
        self.init_params(chanel)

    @property
    def output_size(self):
        return self._batch, self._h, self._w, self._num_filter

    @property
    def state_size(self):
        return self._batch, self._h, self._w, self._num_filter

    def zero_state(self):
        state_size = self.state_size
        return [tf.zeros(state_size, dtype=self._dtype)] * 3

    def init_params(self, chanel):
        """
        init params for convGRU
        Wi: (kernel, kernel, input_chanel, numfilter*3)
        Wh: (kernel, kernel, numfilter, numfilter*3)
        there will be chanel difference between input and state.
        :param chanel: the chanels of input data
        :return:
        """
        self._Wi = tf.get_variable(name=self._name+"_Wi",
                                   shape=(self._k_h, self._k_h,
                                          chanel, self._num_filter*7),
                                   initializer=xavier_initializer(uniform=False),
                                   dtype=self._dtype)
        self._Wh = tf.get_variable(name=self._name + "_Wh",
                                   shape=(self._k_h, self._k_h,
                                          chanel, self._num_filter * 4),
                                   initializer=xavier_initializer(uniform=False),
                                   dtype=self._dtype)
        self._Wm = tf.get_variable(name=self._name + "_Wm",
                                   shape=(self._k_h, self._k_h,
                                          chanel, self._num_filter * 4),
                                   initializer=xavier_initializer(uniform=False),
                                   dtype=self._dtype)
        self._Wc = tf.get_variable(name=self._name + "_Wc",
                                   shape=(self._k_h, self._k_h,
                                          chanel, self._num_filter),
                                   initializer=xavier_initializer(uniform=False),
                                   dtype=self._dtype)
        self._1x1 = tf.get_variable(name=self._name + "_1x1",
                                   shape=(1, 1,
                                          self._num_filter*2, self._num_filter),
                                   initializer=xavier_initializer(uniform=False),
                                   dtype=self._dtype)

        self._Bg = tf.get_variable(name=self._name+"_Bg",
                                   shape=(self._num_filter),
                                   initializer=tf.zeros_initializer,
                                   dtype=self._dtype)
        self._Bi = tf.get_variable(name=self._name + "_Bi",
                                   shape=(self._num_filter),
                                   initializer=tf.zeros_initializer,
                                   dtype=self._dtype)
        self._Bf = tf.get_variable(name=self._name + "_Bf",
                                   shape=(self._num_filter),
                                   initializer=tf.zeros_initializer,
                                   dtype=self._dtype)
        self._Bg_ = tf.get_variable(name=self._name + "_Bg_",
                                   shape=(self._num_filter),
                                   initializer=tf.zeros_initializer,
                                   dtype=self._dtype)
        self._Bi_ = tf.get_variable(name=self._name + "_Bi_",
                                   shape=(self._num_filter),
                                   initializer=tf.zeros_initializer,
                                   dtype=self._dtype)
        self._Bf_ = tf.get_variable(name=self._name + "_Bf_",
                                   shape=(self._num_filter),
                                   initializer=tf.zeros_initializer,
                                   dtype=self._dtype)
        self._Bo = tf.get_variable(name=self._name + "_Bo",
                                   shape=(self._num_filter),
                                   initializer=tf.zeros_initializer,
                                   dtype=self._dtype)

    def __call__(self, inputs, state):
        """
        do a gru computation
        i2h = leakyRelu(Wi*input + bi)  i2h: (b, h, w, 3*filter)
        h2h = leakyRelu(Wh*state + bh)  h2h: (b. h, w, 3*filter)

        :param inputs: tensor (batch, h, w, c)
        :param state: tensor [H, C, M]  each shape is (batch, h, w, c)
        :return:
        """
        if state is None:
            H = self.zero_state()
            C = self.zero_state()
            M = self.zero_state()
        else:
            H, C, M = state

        if inputs is not None:
            i2h = tf.nn.conv2d(inputs,
                               self._Wi,
                               strides=(1, 1, 1, 1),
                               padding="SAME",
                               name=self._name+"_i2h")
            i2h = tf.split(i2h, 7, axis=3)
        else:
            i2h = None

        h2h = tf.nn.conv2d(H,
                           self._Wh,
                           strides=(1, 1, 1, 1),
                           padding="SAME",
                           name=self._name+"_h2h")
        h2h = tf.split(h2h, 4, axis=3)

        m2h = tf.nn.conv2d(M,
                           self._Wm,
                           strides=(1, 1, 1, 1),
                           padding="SAME",
                           name=self._name+"_m2h")
        m2h = tf.split(m2h, 4, axis=3)

        c2h = tf.nn.conv2d(C,
                           self._Wc,
                           strides=(1, 1, 1, 1),
                           padding="SAME",
                           name=self._name+"_1x1")

        if i2h is not None:
            g = tf.nn.tanh(i2h[0] + h2h[0] + self._Bg)
            i = tf.nn.sigmoid(i2h[1] + h2h[1] + self._Bi)
            f = tf.nn.sigmoid(i2h[2] + h2h[2] + self._Bf)

            C = f * C + i * g

            g_ = tf.nn.tanh(i2h[3] + m2h[0] + self._Bg_)
            i_ = tf.nn.sigmoid(i2h[4] + m2h[1] + self._Bi_)
            f_ = tf.nn.sigmoid(i2h[5] + m2h[2] + self._Bf_)

            M = f_ * M + i_ * g_

            o = tf.nn.sigmoid(i2h[6] + m2h[3] + h2h[3] + c2h + self._Bo)
            cm = tf.nn.conv2d(tf.concat([C, M], axis=3),
                              self._1x1, strides=(1, 1, 1, 1),
                              padding="SAME",
                              name=self._name + "_cm")
            H = o * tf.nn.tanh(cm)
        else:
            g = tf.nn.tanh(h2h[0] + self._Bg)
            i = tf.nn.sigmoid(h2h[1] + self._Bi)
            f = tf.nn.sigmoid(h2h[2] + self._Bf)

            C = f * C + i * g

            g_ = tf.nn.tanh(m2h[0] + self._Bg_)
            i_ = tf.nn.sigmoid(m2h[1] + self._Bi_)
            f_ = tf.nn.sigmoid(m2h[2] + self._Bf_)

            M = f_ * M + i_ * g_

            o = tf.nn.sigmoid(m2h[3] + h2h[3] + c2h + self._Bo)
            cm = tf.nn.conv2d(tf.concat([C, M], axis=3),
                              self._1x1, strides=(1, 1, 1, 1),
                              padding="SAME",
                              name=self._name + "_cm")
            H = o * tf.nn.tanh(cm)

        return H, [H, C, M]

    def unroll(self, length, inputs=None, begin_state=None, merge=True):
        """
        Do gru cycle
        :param length: time length
        :param inputs:  (batch, time_seq, H, W, C)
        :param begin_state:
        :param merge: output a list of tensor or a tensor
        :return:
        outputs:
        """
        if begin_state is None:
            states = [self.zero_state()]*3
        else:
            states = begin_state

        outputs = []

        if inputs is not None:
            inputs = tf.unstack(inputs, length, axis=1)
            for i in range(length):
                output, states = self(inputs[i], state=states)
                outputs.append(output)
        else:
            if c.SEQUENCE_MODE:
                inputs = None
                for i in range(length):
                    output, states = self(inputs, state=states)
                    inputs = output
                    outputs.append(output)
            else:
                inputs = [None] * length
                for i in range(length):
                    output, states = self(inputs[i], state=states)
                    outputs.append(output)

        if merge:
            outputs = tf.stack(outputs, axis=1)

        return outputs, states
