import os
import logging
import numpy as np
from PIL import Image

from config import c


def config_log():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s \n %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(c.SAVE_PATH, "train.log"),
                        filemode='w')

    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s  %(levelname)-8s %(message)s')
    # console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)


def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames = new_frames * 3 / 255
    # new_frames /= (255 / 2)
    # new_frames -= 1
    # if frames.min() == np.inf or frames.max() == np.inf or new_frames.min() == np.inf or new_frames.max() == np.inf:
    #     print(frames.min(), frames.max(), new_frames.min(), new_frames.max())
    return new_frames


def denormalize_frames(frames):
    """
    Performs the inverse operation of normalize_frames.

    @param frames: A numpy array. The frames to be converted.

    @return: The denormalized frames.
    """
    new_frames = frames / 3 * 255
    # new_frames = frames + 1
    #new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    new_frames = new_frames.astype(np.uint8)

    return new_frames


def crop_img(img):
    """
    crop tensor
    :param img: (B, T, H, W, C) 
    :return: img (B, T, t_H, t_W, C)
    """
    assert len(img.shape) == 5
    t_H = c.PREDICTION_H
    t_W = c.PREDICTION_W
    *_, H, W, C = img.shape
    if t_H != H or t_W != W:
        side_h = (H - t_H) // 2
        side_w = (W - t_W) // 2
        img = img[:, :, side_h:-side_h, side_w:-side_w, :]
    return img



def save_png(data, path):
    data[data < 0] = 0
    if c.NORMALIZE:
        data = denormalize_frames(data)
    else:
        data = data.astype(np.uint8)

    if not os.path.exists(path):
        os.makedirs(path)
    shape = data.shape
    data = data.reshape(shape[0], shape[-3], shape[-2])
    i = 1
    for img in data[:]:
        img = Image.fromarray(img)
        img.save(os.path.join(path, str(i) + ".png"))
        i += 1
