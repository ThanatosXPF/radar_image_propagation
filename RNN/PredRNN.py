import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from RNN.conv_stlstm import ConvSTLSTMCell

from config import c


class PredRNNCell(object):
    def __init__(self, num_filter, b_h_w,
                 kernel, name, chanel, layers, dtype=tf.float32):
        self._name = name
        self._b_h_w = b_h_w
        self._num_filter = num_filter
        self.chanel = chanel
        self._dtype = dtype
        self._k_h = kernel
        self._layers = layers
        self.cells = []
        self.cell_states = []
        self.stack_rnn_cell()
        self.init_cell_states()

    def stack_rnn_cell(self):
        for i in range(self._layers):
            cell = ConvSTLSTMCell(self._num_filter, self._b_h_w, self._k_h, self._name+f"_cell{i}",
                                  self.chanel, self._dtype)
            self.cells.append(cell)

    def init_cell_states(self):
        for cell in self.cells:
            self.cell_states.append(cell.zero_state())

    def zero_state(self):
        state = []
        for cell in self.cells:
            state.append(cell.zero_state())
        return state

    def __call__(self, inputs, state):
        """
        :param inputs: tensor (batch, h, w, c)
        :param state: tensor [[H, C, M]]  each shape is (batch, h, w, c)
        :return:
        """
        assert len(state) == self._layers
        outputs = []
        for i in range(self._layers):
            if i == 0:
                data = inputs
                M = state[-1][2]

            s = [state[i][0], state[i][1], M]

            _, s = self.cells[i](data, s)
            outputs.append(s)
            M = s[2]
            data = s[0]

        return s[0], outputs

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
