import numpy as np
from model import Model
from config import c


class Deploy(object):
    def __init__(self, load_path=""):
        assert load_path is not None
        self.model = Model(load_path, mode="online")

    def predict(self, in_data):
        gt_data = np.zeros((c.BATCH_SIZE, c.PREDICT_LENGTH, c.H, c.W, c.IN_CHANEL))
        mse, mae, gdl, pred = self.model.valid_step(in_data, gt_data)
        return pred, mse, mae, gdl


if __name__ == '__main__':
    from iterator import Iterator
    it = Iterator(time_interval=["201809161300", "201809161500"],
                  sample_mode="sequent",
                  seq_len=5,
                  stride=1
                  )
    deploy = Deploy("/extend/gru_tf_data/0316_loss_mse/Save/model.ckpt.99999")
    data, *_ = it.sample(batch_size=c.BATCH_SIZE)
    print(data.shape)
    pred, *_ = deploy.predict(data)
    print(pred.min(), pred.max())
    print(pred.shape)
