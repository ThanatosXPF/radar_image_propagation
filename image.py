# Python plugin that supports loading batch of images in parallel
import numpy
import os
from concurrent.futures import ThreadPoolExecutor, wait
import cv2

import config as c


_imread_executor_pool = ThreadPoolExecutor(max_workers=16)


def read_img(path, read_storage, im_h, im_w):
    # read_storage[:] = numpy.fromfile(path, dtype=numpy.uint8).reshape(im_h, im_w)
    read_storage[:] = cv2.imread(path, 0)


def quick_read_frames(path_list, im_h, im_w):
    """Multi-thread Frame Loader

    Parameters
    ----------
    path_list : list
    im_h : height of image
    im_w : width of image

    Returns
    -------

    """
    img_num = len(path_list)
    for i in range(img_num):
        if not os.path.exists(path_list[i]):
            print(path_list[i])
            raise IOError

    read_storage = numpy.empty((img_num, im_h, im_w), dtype=numpy.uint8)

    if img_num == 1:
        read_img(path_list[0], read_storage[0], im_h, im_w)
    else:
        future_objs = []
        for i in range(img_num):
            obj = _imread_executor_pool.submit(read_img, path_list[i], read_storage[i], im_h, im_w)
            future_objs.append(obj)
        wait(future_objs)

    read_storage = read_storage.reshape((img_num, im_h, im_w, c.IN_CHANEL))
    return read_storage[:, ...]
