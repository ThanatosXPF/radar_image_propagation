import numpy as np
import tensorflow as tf

import config as c
from utils import denormalize_frames, normalize_frames


def pixel_to_dBZ(img):
    """

    Parameters
    ----------
    img : np.ndarray or float

    Returns
    -------

    """
    return img * 70.0 - 10.0


def dBZ_to_pixel(dBZ_img):
    """

    Parameters
    ----------
    dBZ_img : np.ndarray

    Returns
    -------

    """
    return np.clip((dBZ_img + 10.0) / 70.0, a_min=0.0, a_max=1.0)


def pixel_to_rainfall(img, a=None, b=None):
    """Convert the pixel values to real rainfall intensity

    Parameters
    ----------
    img : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    if a is None:
        a = c.ZR_a
    if b is None:
        b = c.ZR_b
    dBZ = pixel_to_dBZ(img)
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = np.power(10, dBR / 10.0)
    return rainfall_intensity


def rainfall_to_pixel(rainfall_intensity, a=None, b=None):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    if a is None:
        a = c.ZR_a
    if b is None:
        b = c.ZR_b
    dBR = np.log10(rainfall_intensity) * 10.0
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals

def get_loss_weight_symbol(data):
    if c.USE_BALANCED_LOSS:
        balancing_weights = c.BALANCING_WEIGHTS
        weights = tf.ones_like(data) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in c.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * tf.to_float(data >= threshold)
        weights = weights
    else:
        weights = tf.ones_like(data)
    if c.TEMPORAL_WEIGHT_TYPE == "same":
        return weights
    else:
        raise NotImplementedError


def weighted_mse(pred, gt, weight):
    return weighted_l2(pred, gt, weight)


def weighted_l2(pred, gt, weight):
    """

    Parameters
    ----------
    pred : tensor
        Shape: (batch_size, seq_len, H, W, C)
    gt : tensor
        Shape: (batch_size, seq_len, H, W, C)
    weight : tensor
        Shape: (batch_size, seq_len, H, W, C)

    Returns
    -------
    l2 : Value
    """
    l2 = weight * tf.square(pred - gt)
    l2 = tf.reduce_sum(l2)
    return l2


def weighted_l1(pred, gt, weight):
    l1 = weight * tf.abs(pred - gt)
    l1 = tf.reduce_sum(l1)
    return l1


def weighted_mae(pred, gt, weight):
    return weighted_l1(pred, gt, weight)


def one_step_diff(dat, axis):
    """

    Parameters
    ----------
    dat : tensor (b, length, h, w, c)
    axes : int 2, 3

    Returns
    -------

    """
    if axis == 2:
        return dat[:, :, :-1, :, :] - dat[:, :, 1:, :, :]
    elif axis == 3:
        return dat[:, :, :, :-1, :] - dat[:, :, :, 1:, :]
    else:
        raise NotImplementedError


def gdl_loss(pred, gt):
    """

    Parameters
    ----------
    pred : tensor
        Shape: (b, length, h, w, c)
    gt : tensor
        Shape: (b, length, h, w, c)
    Returns
    -------
    gdl : value
    """
    pred_diff_h = tf.abs(one_step_diff(pred, axis=2))
    pred_diff_w = tf.abs(one_step_diff(pred, axis=3))
    gt_diff_h = tf.abs(one_step_diff(gt, axis=2))
    gt_diff_w = tf.abs(one_step_diff(gt, axis=3))
    gd_h = tf.abs(pred_diff_h - gt_diff_h)
    gd_w = tf.abs(pred_diff_w - gt_diff_w)
    gdl = tf.reduce_sum(gd_h) + tf.reduce_sum(gd_w)
    return gdl


if __name__ == '__main__':
    a= [[1,2,3,5,15],
        [25,30,40,45,50],
        [60, 70, 79, 1, 17],
        [60, 70, 79, 1, 17],
        [60, 70, 79, 1, 17]]
    a = normalize_frames(np.asarray(a))
    print(a)
    print(get_loss_weight_symbol(a))