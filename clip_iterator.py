import os
import numpy as np


class Clip_Iterator(object):
    def __init__(self, data_dir):
        self._data_dir = data_dir

    def sample_clips(self, batch_size):
        clips = []
        targets = np.random.choice(os.listdir(self._data_dir), batch_size)
        for i in range(batch_size):
            path = os.path.join(self._data_dir, targets[i])
            clip = np.load(path)['arr_0']
            if not isinstance(clip, np.ndarray) or len(clip.shape) != 5:
                print(path)
            clips.append(clip)
        try:
            clips = np.concatenate(clips, axis=0)
        except ValueError:
            for c in clips:
                print(c.shape)
        print(clips.shape)
        return clips

    def sample_valid(self, batch):
        targets = os.listdir(self._data_dir)
        clips = []
        for t in targets:
            path = os.path.join(self._data_dir, t)
            clip = np.load(path)['arr_0']
            clips.append(clip)
            if len(clips) == batch:
                yield np.concatenate(clips, axis=0)
                clips = []
