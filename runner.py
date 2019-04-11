import logging
import os
import numpy as np

from model import Model
from iterator import Iterator
from clip_iterator import Clip_Iterator
import config as c
from utils import config_log, save_png
from utils import normalize_frames
from evaluation import Evaluator


class Runner(object):
    def __init__(self, para_tuple=None, mode="train"):

        self.para_tuple = para_tuple

        self.model = Model(para_tuple, mode=mode)
        if not para_tuple:
            self.model.init_params()

    def train(self):
        step = 0
        train_iter = Clip_Iterator(c.TRAIN_DIR_CLIPS)
        while step < c.MAX_ITER:
            data = train_iter.sample_clips(batch_size=c.BATCH_SIZE)
            in_data = data[:, :c.IN_SEQ, ...]

            if c.IN_CHANEL == 3:
                gt_data = data[:, c.IN_SEQ:c.IN_SEQ + c.OUT_SEQ, :, :, 1:-1]
            elif c.IN_CHANEL == 1:
                gt_data = data[:, c.IN_SEQ:c.IN_SEQ + c.OUT_SEQ, ...]
            else:
                raise NotImplementedError

            if c.NORMALIZE:
                in_data = normalize_frames(in_data)
                gt_data = normalize_frames(gt_data)

            mse, mae, gdl = self.model.train_step(in_data, gt_data)
            logging.info(f"Iter {step}: \n\t mse:{mse} \n\t mae:{mae} \n\t gdl:{gdl}")

            if (step + 1) % c.SAVE_ITER == 0:
                self.model.save_model()

            if (step + 1) % c.VALID_ITER == 0:
                self.valid_clips(step)
            step += 1

    def valid_clips(self, step):
        test_iter = Clip_Iterator(c.VALID_DIR_CLIPS)
        evaluator = Evaluator(step)
        i = 0
        for data in test_iter.sample_valid(c.BATCH_SIZE):
            in_data = data[:, :c.IN_SEQ, ...]
            if c.IN_CHANEL == 3:
                gt_data = data[:, c.IN_SEQ:c.IN_SEQ + c.OUT_SEQ, :, :, 1:-1]
            elif c.IN_CHANEL == 1:
                gt_data = data[:, c.IN_SEQ:c.IN_SEQ + c.OUT_SEQ, ...]
            else:
                raise NotImplementedError
            if c.NORMALIZE:
                in_data = normalize_frames(in_data)
                gt_data = normalize_frames(gt_data)

            mse, mae, gdl, pred = self.model.valid_step(in_data, gt_data)
            evaluator.evaluate(gt_data, pred)
            logging.info(f"Iter {step} {i}: \n\t mse:{mse} \n\t mae:{mae} \n\t gdl:{gdl}")
            i += 1
        evaluator.done()

    def run_benchmark(self, iter, mode="Valid"):
        if mode == "Valid":
            time_interval = c.RAINY_VALID
            stride = 20
        else:
            time_interval = c.RAINY_TEST
            stride = 1
        test_iter = Iterator(time_interval=time_interval,
                             sample_mode="sequent",
                             seq_len=c.IN_SEQ + c.OUT_SEQ,
                             stride=1)
        evaluator = Evaluator(iter)
        i = 1
        while not test_iter.use_up:
            data, date_clip, *_ = test_iter.sample(batch_size=c.BATCH_SIZE)
            in_data = np.zeros(shape=(c.BATCH_SIZE, c.IN_SEQ, c.H, c.W, c.IN_CHANEL))
            gt_data = np.zeros(shape=(c.BATCH_SIZE, c.OUT_SEQ, c.H, c.W, 1))
            if type(data) == type([]):
                break
            in_data[...] = data[:, :c.IN_SEQ, ...]

            if c.IN_CHANEL == 3:
                gt_data[...] = data[:, c.IN_SEQ:c.IN_SEQ + c.OUT_SEQ, :, :, 1:-1]
            elif c.IN_CHANEL == 1:
                gt_data[...] = data[:, c.IN_SEQ:c.IN_SEQ + c.OUT_SEQ, ...]
            else:
                raise NotImplementedError

            # in_date = date_clip[0][:c.IN_SEQ]

            if c.NORMALIZE:
                in_data = normalize_frames(in_data)
                gt_data = normalize_frames(gt_data)

            mse, mae, gdl, pred = self.model.valid_step(in_data, gt_data)
            evaluator.evaluate(gt_data, pred)
            logging.info(f"Iter {iter} {i}: \n\t mse:{mse} \n\t mae:{mae} \n\t gdl:{gdl}")
            i += 1
            if i % stride == 0:
                if c.IN_CHANEL == 3:
                    in_data = in_data[:, :, :, :, 1:-1]

                for b in range(c.BATCH_SIZE):
                    predict_date = date_clip[b][c.IN_SEQ]
                    logging.info(f"Save {predict_date} results")
                    if mode == "Valid":
                        save_path = os.path.join(c.SAVE_VALID, str(iter), predict_date.strftime("%Y%m%d%H%M"))
                    else:
                        save_path = os.path.join(c.SAVE_TEST, str(iter), predict_date.strftime("%Y%m%d%H%M"))

                    path = os.path.join(save_path, "in")
                    save_png(in_data[b], path)

                    path = os.path.join(save_path, "pred")
                    save_png(pred[b], path)

                    path = os.path.join(save_path, "out")
                    save_png(gt_data[b], path)
        evaluator.done()


def test(para, iter, mode="Test"):
    model = Runner(para, mode=mode)
    model.run_benchmark(iter, mode=mode)


if __name__ == '__main__':
    config_log()
    # paras = ("first_try", "94999")
    paras = None
    runner = Runner(paras)
    runner.train()
