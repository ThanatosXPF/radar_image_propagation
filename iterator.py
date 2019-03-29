import os
import pandas as pd
import numpy as np
import bisect

import config as cfg
from image import quick_read_frames


class Iterator(object):
    """The iterator for the dataset

    """

    def __init__(self, time_interval, sample_mode, seq_len=30,
                 max_consecutive_missing=0, begin_ind=None, end_ind=None,
                 stride=None, width=None, height=None, base_freq='6min'):
        """Random sample: sample a random clip that will not violate the max_missing frame_num criteria
        Sequent sample: sample a clip from the beginning of the time.
                        Everytime, the clips from {T_begin, T_begin + 6min, ..., T_begin + (seq_len-1) * 6min} will be used
                        The begin datetime will move forward by adding stride: T_begin += 6min * stride
                        Once the clips violates the maximum missing number criteria, the starting
                         point will be moved to the next datetime that does not violate the missing_frame criteria

        Parameters
        ----------
        time_interval : list
            path of the saved pandas dataframe
        sample_mode : str
            Can be "random" or "sequent"
        seq_len : int
        max_consecutive_missing : int
            The maximum consecutive missing frames
        begin_ind : int
            Index of the begin frame
        end_ind : int
            Index of the end frame
        stride : int or None, optional
        width : int or None, optional
        height : int or None, optional
        base_freq : str, optional
        """
        assert isinstance(time_interval, list)
        self.time_interval = time_interval
        if width is None:
            width = cfg.W
        if height is None:
            height = cfg.H

        self._df = self._df_generate()
        print("df size {}".format(self._df.size))

        self.set_begin_end(begin_ind=begin_ind, end_ind=end_ind)
        self._df_index_set = frozenset([self._df.index[i] for i in range(self._df.size)])
        self._seq_len = seq_len
        self._width = width
        self._height = height
        self._stride = stride
        self._max_consecutive_missing = max_consecutive_missing
        self._base_freq = base_freq
        self._base_time_delta = pd.Timedelta(base_freq)
        assert sample_mode in ["random", "sequent"], "Sample mode=%s is not supported" % sample_mode
        self.sample_mode = sample_mode
        if sample_mode == "sequent":
            assert self._stride is not None
            self._current_datetime = self.begin_time
            self._buffer_mult = 6
            self._buffer_datetime_keys = None
            self._buffer_frame_dat = None
            self._buffer_mask_dat = None
        else:
            self._max_buffer_length = None

    def set_begin_end(self, begin_ind=None, end_ind=None):
        self._begin_ind = 0 if begin_ind is None else begin_ind
        self._end_ind = self.total_frame_num - 1 if end_ind is None else end_ind

    @property
    def total_frame_num(self):
        return self._df.size

    @property
    def begin_time(self):
        return self._df.index[self._begin_ind]

    @property
    def end_time(self):
        return self._df.index[self._end_ind]

    @property
    def use_up(self):
        if self.sample_mode == "random":
            return False
        else:
            return self._current_datetime > self.end_time

    def _get_df(self):
        ref_path = cfg.REF_PATH
        refs = os.listdir(ref_path)
        refs = sorted(refs)
        date_list = []
        for file_ in refs:
            date = file_.split("_")[2]
            date = pd.to_datetime(date)
            date_list.append(date)
        df = pd.DataFrame([1 for i in range(len(date_list))],
                          columns=["rain"], index=date_list)
        return df

    def _df_generate(self):
        df = self._get_df()
        begin, end = self.time_interval
        begin = pd.to_datetime(begin)
        end = pd.to_datetime(end)
        date_list = []
        for date in df.index:
            if end >= date >= begin:
                date_list.append(date)
        new_df = pd.DataFrame([1 for i in range(len(date_list))],
                              columns=["rain"], index=date_list)
        return new_df

    def _next_exist_timestamp(self, timestamp):
        next_ind = bisect.bisect_right(self._df.index, timestamp)
        if next_ind >= self._df.size:
            return None
        else:
            return self._df.index[bisect.bisect_right(self._df.index, timestamp)]

    def _is_valid_clip(self, datetime_clip):
        """Check if the given datetime_clip is valid

        Parameters
        ----------
        datetime_clip :

        Returns
        -------
        ret : bool
        """
        missing_count = 0
        for i in range(len(datetime_clip)):
            if datetime_clip[i] not in self._df_index_set:
                return False
        return True

    def _load_frames(self, datetime_clips):
        assert isinstance(datetime_clips, list)
        for clip in datetime_clips:
            assert len(clip) == self._seq_len
        batch_size = len(datetime_clips)
        frame_dat = np.zeros(( batch_size, self._seq_len, self._height,
                               self._width, cfg.IN_CHANEL),
                             dtype=np.uint8)

        if self.sample_mode == "random":
            paths = []
            hit_inds = []
            miss_inds = []
            for i in range(self._seq_len):
                for j in range(batch_size):
                    timestamp = datetime_clips[j][i]
                    if timestamp in self._df_index_set:
                        paths.append(convert_datetime_to_filepath(datetime_clips[j][i]))
                        hit_inds.append([i, j])
                    else:
                        miss_inds.append([i, j])
            hit_inds = np.array(hit_inds, dtype=np.int)
            all_frame_dat = quick_read_frames(paths, self._height, self._width)
            frame_dat[hit_inds[:, 1], hit_inds[:, 0], :, :, :] = all_frame_dat
        else:
            # Get the first_timestamp and the last_timestamp in the datetime_clips
            first_timestamp = datetime_clips[-1][-1]
            last_timestamp = datetime_clips[0][0]
            for i in range(self._seq_len):
                for j in range(batch_size):
                    timestamp = datetime_clips[j][i]
                    if timestamp in self._df_index_set:
                        first_timestamp = min(first_timestamp, timestamp)
                        last_timestamp = max(last_timestamp, timestamp)
            if self._buffer_datetime_keys is None or \
                    not (first_timestamp in self._buffer_datetime_keys
                         and last_timestamp in self._buffer_datetime_keys):
                read_begin_ind = self._df.index.get_loc(first_timestamp)
                read_end_ind = self._df.index.get_loc(last_timestamp) + 1
                read_end_ind = min(read_begin_ind +
                                   self._buffer_mult * (read_end_ind - read_begin_ind),
                                   self._df.size)
                self._buffer_datetime_keys = self._df.index[read_begin_ind:read_end_ind]
                # Fill in the buffer
                paths = []
                for i in range(self._buffer_datetime_keys.size):
                    paths.append(convert_datetime_to_filepath(self._buffer_datetime_keys[i]))
                self._buffer_frame_dat = quick_read_frames(paths, self._height, self._width)
            for i in range(self._seq_len):
                for j in range(batch_size):
                    timestamp = datetime_clips[j][i]
                    if timestamp in self._df_index_set:
                        assert timestamp in self._buffer_datetime_keys
                        ind = self._buffer_datetime_keys.get_loc(timestamp)
                        frame_dat[j, i, :, :, :] = self._buffer_frame_dat[ind, :, :, :]
        return frame_dat

    def reset(self, begin_ind=None, end_ind=None):
        assert self.sample_mode == "sequent"
        self.set_begin_end(begin_ind=begin_ind, end_ind=end_ind)
        self._current_datetime = self.begin_time

    def random_reset(self):
        assert self.sample_mode == "sequent"
        self.set_begin_end(begin_ind=np.random.randint(0,
                                                       self.total_frame_num -
                                                       5 * self._seq_len),
                           end_ind=None)
        self._current_datetime = self.begin_time

    def check_new_start(self):
        assert self.sample_mode == "sequent"
        datetime_clip = pd.date_range(start=self._current_datetime,
                                      periods=self._seq_len,
                                      freq=self._base_freq)
        if self._is_valid_clip(datetime_clip):
            return self._current_datetime == self.begin_time
        else:
            return True

    def sample(self, batch_size, only_return_datetime=False):
        """Sample a minibatch from the sz radar ref dataset based on the given type and pd_file

        Parameters
        ----------
        batch_size : int
            Batch size
        only_return_datetime : bool
            Whether to only return the datetimes
        Returns
        -------
        frame_dat : np.ndarray
            Shape: (seq_len, valid_batch_size, 1, height, width)
        mask_dat : np.ndarray
            Shape: (seq_len, valid_batch_size, 1, height, width)
        datetime_clips : list
            length should be valid_batch_size
        new_start : bool
        """
        if self.sample_mode == 'sequent':
            if self.use_up:
                raise ValueError("The Iterator has been used up!")
            datetime_clips = []
            new_start = False
            for i in range(batch_size):
                while not self.use_up:
                    datetime_clip = pd.date_range(start=self._current_datetime,
                                                  periods=self._seq_len,
                                                  freq=self._base_freq)
                    if self._is_valid_clip(datetime_clip):
                        new_start = new_start or (self._current_datetime == self.begin_time)
                        datetime_clips.append(datetime_clip)
                        self._current_datetime += self._stride * self._base_time_delta
                        # self._current_datetime += self._base_time_delta
                        break
                    else:
                        new_start = True
                        self._current_datetime = \
                            self._next_exist_timestamp(timestamp=self._current_datetime)
                        if self._current_datetime is None:
                            # This indicates that there is no timestamp left,
                            # We point the current_datetime to be the next timestamp of self.end_time
                            self._current_datetime = self.end_time + self._base_time_delta
                            break
                        continue
            new_start = None if batch_size != 1 else new_start
            if only_return_datetime:
                return datetime_clips, new_start
            else:
                if self.use_up:
                    return [], [], False, True
                else:
                    frame_dat = self._load_frames(datetime_clips=datetime_clips)
                    return frame_dat, datetime_clips, new_start, False

        else:
            assert only_return_datetime is False
            datetime_clips = []
            new_start = None
            for i in range(batch_size):
                while True:
                    rand_ind = np.random.randint(0, self._df.size, 1)[0]
                    random_datetime = self._df.index[rand_ind]
                    datetime_clip = pd.date_range(start=random_datetime,
                                                  periods=self._seq_len,
                                                  freq=self._base_freq)
                    if self._is_valid_clip(datetime_clip):
                        datetime_clips.append(datetime_clip)
                        break
        if not datetime_clips:
            return [], [], [], False
        frame_dat = self._load_frames(datetime_clips=datetime_clips)
        return frame_dat, datetime_clips, new_start


def convert_datetime_to_filepath(date_time):
    """Convert datetime to the filepath

    Parameters
    ----------
    date_time : datetime.datetime

    Returns
    -------
    ret : str
    """
    # ret = os.path.join(cfg.REF_PATH, "cappi_ref_"+"%04d%02d%02d%02d%02d" %(
    #     date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute)
    #     +"_2500_0.ref")
    ret = os.path.join(cfg.REF_PATH, "cappi_ref_" + date_time.strftime("%Y%m%d%H%M")
                       + "_2500_0.png")
    return ret