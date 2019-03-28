import os
import pandas as pd
import bisect

import config as c


class Iterator(object):
    """The iterator for the dataset

    """

    def __init__(self, time_interval, sample_mode, seq_len=30,
                 begin_ind=None, end_ind=None,
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
            width = c.W
        if height is None:
            height = c.H

        self._df = self._df_generate()
        print("df size {}".format(self._df.size))

        self.set_begin_end(begin_ind=begin_ind, end_ind=end_ind)
        self._df_index_set = frozenset([self._df.index[i] for i in range(self._df.size)])
        self._seq_len = seq_len
        self._width = width
        self._height = height
        self._stride = stride

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
        ref_path = c.REF_PATH
        refs = os.listdir(ref_path)
        refs = sorted(refs)
        date_list = []
        for file_ in refs:
            date = file_.split("_")[2]
            date = pd.to_datetime(date)
            date_list.append(date)
        df = pd.DataFrame([1] * len(date_list),
                          columns=["rain"], index=date_list)
        return df

    def _df_generate(self):
        begin, end = self.time_interval
        begin = pd.to_datetime(begin)
        end = pd.to_datetime(end)

        ref_path = c.REF_PATH
        refs = os.listdir(ref_path)
        refs = sorted(refs)
        date_list = []
        for file_ in refs:
            date = file_.split("_")[2]
            date = pd.to_datetime(date)
            if begin <= date <= end:
                date_list.append(date)

        new_df = pd.DataFrame([1] * len(date_list),
                              columns=["rain"], index=date_list)
        return new_df

    def _next_exist_timestamp(self, timestamp):
        next_ind = bisect.bisect_right(self._df.index, timestamp)
        if next_ind >= self._df.size:
            return None
        else:
            return self._df.index[bisect.bisect_right(self._df.index, timestamp)]