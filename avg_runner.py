import os
import logging
import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator
from clip_iterator import Clip_Iterator
from iterator import Iterator
from utils import normalize_frames, config_log, save_png
from evaluation import Evaluator
from config import c, cfg_from_file


class AVGRunner:
    def __init__(self, restore_path, mode="train"):
        self.global_step = 0
        self.num_steps = c.MAX_ITER

        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.summary_writer = tf.summary.FileWriter(c.SAVE_SUMMARY, graph=self.sess.graph)

        if mode == "train":
            self._out_seq = c.OUT_SEQ
            self._h = c.H
            self._w = c.W
        else:
            # self._batch = 1
            self._out_seq = c.PREDICT_LENGTH
            self._h = c.PREDICTION_H
            self._w = c.PREDICTION_W

        self._in_seq = c.IN_SEQ
        self._batch = c.BATCH_SIZE

        self.g_model = Generator(self.sess, self.summary_writer, mode=mode)
        if c.ADVERSARIAL and mode == "train":
            self.d_model = Discriminator(self.sess, self.summary_writer)
        else:
            self.d_model = None

        self.saver = tf.train.Saver(max_to_keep=0)

        if restore_path is not None:
            self.saver.restore(self.sess, restore_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    def get_train_batch(self, iterator):
        data, *_ = iterator.sample(batch_size=self._batch)
        in_data = data[:, :self._in_seq, :, :, :]

        if c.IN_CHANEL == 3:
            gt_data = data[:, self._in_seq:self._in_seq + self._out_seq, :, :, :]
        elif c.IN_CHANEL == 1:
            gt_data = data[:, self._in_seq:self._in_seq + self._out_seq, :, :,:]
        else:
            raise NotImplementedError

        if c.NORMALIZE:
            in_data = normalize_frames(in_data)
            gt_data = normalize_frames(gt_data)
        return in_data, gt_data

    def train(self):
        train_iter = Iterator(time_interval=c.RAINY_TRAIN,
                             sample_mode="random",
                             seq_len=self._in_seq + self._out_seq,
                             stride=1)
        while self.global_step < c.MAX_ITER:

            in_data, gt_data = self.get_train_batch(train_iter)

            if c.ADVERSARIAL and self.global_step > c.ADV_INVOLVE:
                print("start d_model")
                d_loss, *_ = self.d_model.train_step(in_data, gt_data, self.g_model)
            else:
                d_loss = 0

            in_data, gt_data = self.get_train_batch(train_iter)

            g_loss, mse, gd_loss, global_step = self.g_model.train_step(in_data, gt_data, self.d_model)

            self.global_step = global_step

            logging.info(f"Iter {self.global_step}: \n\t "
                         f"g_loss: {g_loss:.4f} \n\t"
                         f"mse: {mse:.4f} \n\t "
                         f"mse_real: {gd_loss:.4f} \n\t"
                         f"d_loss: {d_loss:.4f}")

            if (self.global_step + 1) % c.SAVE_ITER == 0:
                self.save_model()

            if (self.global_step + 1) % c.VALID_ITER == 0:
                self.run_benchmark(global_step, mode="Valid")

    def valid(self):
        test_iter = Clip_Iterator(c.VALID_DIR_CLIPS)
        evaluator = Evaluator(self.global_step)
        i = 0
        for data in test_iter.sample_valid(self._batch):
            in_data = data[:, :self._in_seq, ...]
            if c.IN_CHANEL == 3:
                gt_data = data[:, self._in_seq:self._in_seq + self._out_seq, :, :, 1:-1]
            elif c.IN_CHANEL == 1:
                gt_data = data[:, self._in_seq:self._in_seq + self._out_seq, ...]
            else:
                raise NotImplementedError
            if c.NORMALIZE:
                in_data = normalize_frames(in_data)
                gt_data = normalize_frames(gt_data)

            mse, mae, gdl, pred = self.g_model.valid_step(in_data, gt_data)
            evaluator.evaluate(gt_data, pred)
            logging.info(f"Iter {self.global_step} {i}: \n\t "
                         f"mse:{mse:.4f} \n\t "
                         f"mae:{mae:.4f} \n\t "
                         f"gdl:{gdl:.4f}")
            i += 1
        evaluator.done()

    def save_model(self):
        from os.path import join
        save_path = self.saver.save(self.sess,
                                    join(c.SAVE_MODEL, "model.ckpt"),
                                    global_step=self.global_step)
        print("Model saved in path: %s" % save_path)

    def run_benchmark(self, iter, mode="Test"):
        if mode == "Valid":
            time_interval = c.RAINY_VALID
            stride = 5
        else:
            time_interval = c.RAINY_TEST
            stride = 1
        test_iter = Iterator(time_interval=time_interval,
                             sample_mode="sequent",
                             seq_len=self._in_seq + self._out_seq,
                             stride=1)
        evaluator = Evaluator(iter, length=self._out_seq, mode=mode)
        i = 1
        while not test_iter.use_up:
            data, date_clip, *_ = test_iter.sample(batch_size=self._batch)
            in_data = np.zeros(shape=(self._batch, self._in_seq, self._h, self._w, c.IN_CHANEL))
            gt_data = np.zeros(shape=(self._batch, self._out_seq, self._h, self._w, 1))
            if type(data) == type([]):
                break
            in_data[...] = data[:, :self._in_seq, :, :, :]

            if c.IN_CHANEL == 3:
                gt_data[...] = data[:, self._in_seq:self._in_seq + self._out_seq, :, :, :]
            elif c.IN_CHANEL == 1:
                gt_data[...] = data[:, self._in_seq:self._in_seq + self._out_seq, :, :, :]
            else:
                raise NotImplementedError

            # in_date = date_clip[0][:c.IN_SEQ]

            if c.NORMALIZE:
                in_data = normalize_frames(in_data)
                gt_data = normalize_frames(gt_data)

            mse, mae, gdl, pred = self.g_model.valid_step(in_data, gt_data)
            evaluator.evaluate(gt_data, pred)
            logging.info(f"Iter {iter} {i}: \n\t mse:{mse} \n\t mae:{mae} \n\t gdl:{gdl}")
            i += 1
            if i % stride == 0:
                if c.IN_CHANEL == 3:
                    in_data = in_data[:, :, :, :, 1:-1]

                for b in range(self._batch):
                    predict_date = date_clip[b][self._in_seq-1]
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
    model = AVGRunner(para, mode)
    model.run_benchmark(iter, mode=mode)


if __name__ == '__main__':
    # config_log()
    # cfg_from_file("/extend/gru_tf_data/0606_ebgan/cfg0.yml")
    # paras = ("first_try", "94999")
    # paras = '/extend/gru_tf_data/0512_ebtest/Save/model.ckpt-49999'
    paras = None
    runner = AVGRunner(paras)
    runner.train()
    # test(paras, 49999)



