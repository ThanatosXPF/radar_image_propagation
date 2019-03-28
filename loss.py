import tensorflow as tf

from evaluation import get_loss_weight_symbol, \
    weighted_mse, weighted_mae, gdl_loss
import config as c


class Loss(object):
    def __init__(self):
        self._batch = c.BATCH_SIZE
        self._in_seq = c.IN_SEQ
        self._out_seq = c.OUT_SEQ
        self._h = c.H
        self._w = c.W
        self._in_c = c.IN_CHANEL

    def loss_sym(self, pred, gt):
        weights = get_loss_weight_symbol(pred)
        mse = weighted_mse(pred, gt, weights)
        mae = weighted_mae(pred, gt, weights)
        gdl = gdl_loss(pred, gt)
        return mse, mae, gdl

