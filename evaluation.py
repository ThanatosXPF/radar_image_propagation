import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
plt.switch_backend('agg')

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


class Evaluator(object):
    def __init__(self, step):
        self.metric = {}
        for threshold in c.EVALUATION_THRESHOLDS:
            self.metric[threshold] = {
                "pod": np.zeros((c.OUT_SEQ, 1), np.float32),
                "far": np.zeros((c.OUT_SEQ, 1), np.float32),
                "csi": np.zeros((c.OUT_SEQ, 1), np.float32),
                "hss": np.zeros((c.OUT_SEQ, 1), np.float32)
            }
        self.step = step
        self.total = 0
        print(self.metric.keys())

    def get_metrics(self, gt, pred, threshold):
        b_gt = gt > threshold
        b_pred = pred > threshold
        b_gt_n = np.logical_not(b_gt)
        b_pred_n = np.logical_not(b_pred)

        summation_axis = (0, 2, 3)

        hits = np.logical_and(b_pred, b_gt).sum(axis=summation_axis)
        misses = np.logical_and(b_pred_n, b_gt).sum(axis=summation_axis)
        false_alarms = np.logical_and(b_pred, b_gt_n).sum(axis=summation_axis)
        correct_negatives = np.logical_and(b_pred_n, b_gt_n).sum(axis=summation_axis)

        a = hits
        b = false_alarms
        c = misses
        d = correct_negatives

        pod = a / (a + c)
        far = b / (a + b)
        csi = a / (a + b + c)
        n = a + b + c + d
        aref = (a + b) / n * (a + c)
        gss = (a - aref) / (a + b + c - aref)
        hss = 2 * gss / (gss + 1)
        return pod, far, csi, hss

    def evaluate(self, gt, pred):
        self.total += 1
        for threshold in c.EVALUATION_THRESHOLDS:
            pod, far, csi, hss = self.get_metrics(gt, pred, threshold)
            self.metric[threshold]["pod"] += pod
            self.metric[threshold]["far"] += far
            self.metric[threshold]["csi"] += csi
            self.metric[threshold]["hss"] += hss

    def done(self):
        thresholds = c.EVALUATION_THRESHOLDS
        pods = []
        fars = []
        csis = []
        hsss = []
        save_path = join(c.SAVE_METRIC, f"{self.step}")
        if not exists(save_path):
            makedirs(save_path)
        # draw line chart
        for threshold in thresholds:
            metrics = self.metric[threshold]
            pod = metrics["pod"].reshape(-1) / self.total
            pods.append(np.average(pod))
            far = metrics["far"].reshape(-1) / self.total
            fars.append(np.average(far))
            csi = metrics["csi"].reshape(-1) / self.total
            csis.append(np.average(csi))
            hss = metrics["hss"].reshape(-1) / self.total
            hsss.append(np.average(hss))

            x = list(range(len(pod)))
            plt.plot(x, pod, "r--", label='pod')
            plt.plot(x, far, "g--", label="far")
            plt.plot(x, csi, "b--", label="csi")
            plt.plot(x, hss, "k--", label="hss")
            for a, p, f, cs, h in zip(x, pod, far, csi, hss):
                plt.text(a, p+0.005, "%.2f"%p, ha='center', va='bottom', fontsize=7)
                plt.text(a, f+0.005, "%.2f"%f, ha='center', va='bottom', fontsize=7)
                plt.text(a, cs+0.005, "%.2f"%cs, ha='center', va='bottom', fontsize=7)
                plt.text(a, h+0.005, "%.2f"%h, ha='center', va='bottom', fontsize=7)

            plt.title(f"Threshold {threshold}")
            plt.xlabel("Time step")
            plt.ylabel("Rate")
            plt.legend()
            plt.gcf().set_size_inches(9.6, 4.8)
            plt.savefig(join(save_path, f"{threshold}.jpg"))
            plt.clf()
        # draw bar chart
        x = np.array(range(len(thresholds)))
        total_width, n = 0.8, 4
        width = total_width / n
        plt.bar(x, pods, width=width, label='pod', fc='r')
        plt.bar(x+0.2, fars, width=width, label='far', fc='g', tick_label=thresholds)
        plt.bar(x+0.4, csis, width=width, label='csi', fc='b')
        plt.bar(x+0.6, hsss, width=width, label='hss', fc='k')
        for a, p, f, cs, h in zip(x, pods, fars, csis, hsss):
            plt.text(a, p + 0.005, "%.2f" % p, ha='center', va='bottom', fontsize=7)
            plt.text(a+0.2, f + 0.005, "%.2f" % f, ha='center', va='bottom', fontsize=7)
            plt.text(a+0.4, cs + 0.005, "%.2f" % cs, ha='center', va='bottom', fontsize=7)
            plt.text(a+0.6, h + 0.005, "%.2f" % h, ha='center', va='bottom', fontsize=7)
        plt.xlabel("Thresholds")
        plt.ylabel("Rate")
        plt.title(f"Average metrics in {self.step}")
        plt.legend()
        plt.gcf().set_size_inches(9.6, 4.8)
        plt.savefig(join(save_path, f"average_{self.step}.jpg"))
        plt.clf()


if __name__ == '__main__':
    from cv2 import imread
    e = Evaluator(100)
    gt = np.zeros((1, 10, 900, 900, 1))
    pred = np.zeros((1, 10, 900, 900, 1))
    for i in range(1, 11):
        gt[:, i-1, :,:,0] = imread(f'/extend/results/gru_tf/3_99999_h/20180319222400/out/{i}.png', 0)
        pred[:, i-1, :,:,0] = imread(f'/extend/results/gru_tf/3_99999_h/20180319222400/pred/{i}.png', 0)

    e.evaluate(gt, pred)
    for i in range(1, 11):
        gt[:, i-1, :,:,0] = imread(f'/extend/results/gru_tf/3_99999_h/20180319232400/out/{i}.png', 0)
        pred[:, i-1, :,:,0] = imread(f'/extend/results/gru_tf/3_99999_h/20180319232400/pred/{i}.png', 0)
    e.evaluate(gt, pred)
    e.done()