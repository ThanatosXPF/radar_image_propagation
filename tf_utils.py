import numpy as np
import tensorflow as tf

from config import c


def conv2d_act(input, name, kernel, bias, strides, act_type="leaky", padding="SAME"):
    """
    :param input:
    :param name:
    :param kernel:
    :param bias:
    :param strides:
    :param act_type:
    :param padding:
    :return:
    """
    with tf.name_scope(name):
        input_size = input.shape.as_list()
        if len(input_size) == 5:
            input = tf.reshape(input, shape=(input_size[0] * input_size[1],
                                             input_size[2],
                                             input_size[3],
                                             input_size[4]))
        if c.DOWN_SAMPLE_TYPE == "conv":
            out = conv2d(input=input, kernel=kernel, bias=bias, strides=(1, strides, strides, 1),
                         act_type=act_type, padding=padding)
        elif c.DOWN_SAMPLE_TYPE == "inception":
            out = inception_conv_2d(input=input, kernels=kernel, biases=bias, strides=(1, strides, strides, 1),
                         act_type=act_type, padding=padding)
        else:
            raise NotImplementedError
        if len(input_size) == 5:
            out_size = out.shape.as_list()
            out = tf.reshape(out, shape=(input_size[0],
                                         input_size[1],
                                         out_size[-3],
                                         out_size[-2],
                                         out_size[-1]))
        return out


def conv2d(input, kernel, bias, strides, act_type="leaky", padding="SAME"):
    out = tf.nn.conv2d(input, kernel, strides=strides, padding=padding)
    out = tf.nn.bias_add(out, bias)
    if act_type == "relu":
        out = tf.nn.relu(out)
    elif act_type == "leaky":
        out = tf.nn.leaky_relu(out, alpha=0.2)
    return out


def inception_conv_2d(input, kernels, biases, strides, act_type="leaky", padding="SAME"):
    result = []
    for kernel, bias in zip(kernels, biases):
        out = conv2d(input, kernel, bias, strides, act_type, padding)
        result.append(out)
    result = tf.concat(result, axis=-1)
    return result


def deconv2d_act(input, name, kernel, bias, infer_shape, strides, act_type="leaky", padding="SAME"):
    with tf.name_scope(name):
        input_size = input.shape.as_list()
        if len(input_size) == 5:
            input = tf.reshape(input, shape=(input_size[0] * input_size[1],
                                             input_size[2],
                                             input_size[3],
                                             input_size[4]))

        if c.UP_SAMPLE_TYPE == "deconv":
            out = deconv2d(input=input, kernel=kernel, bias=bias, infer_shape=infer_shape,
                           strides=(1, strides, strides, 1), padding=padding, act_type=act_type)
        elif c.UP_SAMPLE_TYPE == "inception":
            out = inception_deconv_2d(input=input, kernels=kernel, biases=bias, infer_shapes=infer_shape,
                           strides=(1, strides, strides, 1), padding=padding, act_type=act_type)
        else:
            raise NotImplementedError
        if len(input_size) == 5:
            out_size = out.shape.as_list()
            out = tf.reshape(out, shape=(input_size[0],
                                         input_size[1],
                                         out_size[-3],
                                         out_size[-2],
                                         out_size[-1]))
        return out


def deconv2d(input, kernel, bias, infer_shape, strides, padding, act_type):
    out = tf.nn.conv2d_transpose(input, kernel, infer_shape, strides=strides,
                                 padding=padding)
    out = tf.nn.bias_add(out, bias)
    if act_type == "relu":
        out = tf.nn.relu(out)
    elif act_type == "leaky":
        out = tf.nn.leaky_relu(out, alpha=0.2)
    return out


def inception_deconv_2d(input, kernels, biases, infer_shapes, strides, act_type="leaky", padding="SAME"):
    result = []
    for kernel, bias, infer_shape in zip(kernels, biases, infer_shapes):
        out = deconv2d(input, kernel, bias, infer_shape, strides, padding, act_type)
        result.append(out)
    result = tf.concat(result, axis=-1)
    return result


def downsample(x, name, kshape=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
    with tf.name_scope(name):
        out = tf.nn.max_pool(x,
                             ksize=kshape, #size of window
                             strides=strides,
                             padding='SAME')
        return out


def upsample(input, name, factor=(2, 2)):
    size = [int(input.shape[-3] * factor[0]), int(input.shape[-2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out


def fullyConnected(input, name, output_size, dtype=np.float32):
    with tf.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.get_variable(name='w_'+name,
                            shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            dtype=dtype)
        b = tf.get_variable(name='b_'+name,
                            shape=[output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                            dtype=dtype)
        input = tf.reshape(input, [-1, input_size])
        out = tf.nn.relu(tf.add(tf.matmul(input, W), b))
        return out


def dropout(input, name, keep_rate):
    with tf.name_scope(name):
        out = tf.nn.dropout(input, keep_rate)
        return out


def weighted_l2(pred, gt):
    weight = get_loss_weight_symbol(gt)
    l2 = weight * tf.square(pred - gt)
    l2 = tf.reduce_sum(l2)
    return l2


def get_loss_weight_symbol(data):
    balancing_weights = c.BALANCING_WEIGHTS
    thresholds = c.THRESHOLDS
    weights = tf.zeros_like(data)
    if c.USE_BALANCED_LOSS:
        for i in range(len(thresholds)):
            weights = weights + balancing_weights[i] * tf.to_float(data >= thresholds[i])
    return weights




if __name__ == '__main__':
    gt = np.random.rand(5,5) * 255
    gt = gt.astype(np.uint8)
    print(gt)
    print(get_loss_weight_symbol(gt))