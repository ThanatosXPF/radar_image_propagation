import os
import numpy as np
from glob import glob

import config as c


def get_a_process(data_dir,  seq_len, height, width, chanel=1):
    clips = np.empty(shape=[1, seq_len, height, width, chanel], dtype=np.uint8)
    ep_dir = os.path.join(data_dir, np.random.choice(os.listdir(data_dir), 1)[0])
    ep_frames = sorted(os.listdir(ep_dir))
    start_index = np.random.choice(len(ep_frames) - seq_len + 1)
    clip_frame_paths = ep_frames[start_index: start_index + seq_len]
    for frame_num, frame_name in enumerate(clip_frame_paths):
        frame_path = os.path.join(ep_dir, frame_name)
        try:
            frame = np.fromfile(frame_path, dtype=np.uint8).reshape(height, width, chanel)
        except ValueError:
            print(frame_path)
        else:
            frame[frame > 80] = 0
            frame[frame < 15] = 0
            clips[0, frame_num, ...] = frame[:,:,:]
    return clips


def process_clip():
    while True:
        clip = get_a_process("/extend/AVG_data/TrainingData/14-17_2500", c.IN_SEQ+c.OUT_SEQ, c.FULL_H, c.FULL_W)
        take_first = np.random.choice(2, p=[0.95, 0.05])
        cropped_clip = np.empty([1,  c.IN_SEQ+c.OUT_SEQ, c.H, c.W, 1])
        for i in range(100):
            crop_x = np.random.choice(c.FULL_W - c.W + 1)
            crop_y = np.random.choice(c.FULL_H - c.H + 1)
            cropped_clip[...] = clip[:, :, crop_y:crop_y+c.H, crop_x:crop_x+c.W, :]
            if take_first or np.sum(cropped_clip > 0) > (c.MOVEMENT_THRESHOLD * (c.IN_SEQ+c.OUT_SEQ)):
                if not isinstance(cropped_clip, np.ndarray) or cropped_clip.shape != (1, c.IN_SEQ+c.OUT_SEQ, c.H, c.W, 1):
                    print( crop_x, crop_y, clip.shape)
                return cropped_clip


def process_training_data(num_clips):
    num_prev_clips = len(glob(c.TRAIN_DIR_CLIPS + '*'))

    for clip_num in range(num_prev_clips, num_clips + num_prev_clips):
        clip = process_clip()

        if not isinstance(clip, np.ndarray) or clip.shape != (1, c.IN_SEQ+c.OUT_SEQ, c.H, c.W, 1):
            print("fuck", type(clip))
            continue
        elif clip.max() == 0:
            print("All 0 warning!!!")
        np.savez_compressed(os.path.join(c.TRAIN_DIR_CLIPS, str(clip_num)), clip)

        if (clip_num + 1) % 100 == 0: print('Processed %d clips' % (clip_num + 1))


def main():
    num_clips = 10000

    process_training_data(num_clips)


if __name__ == '__main__':
    main()
