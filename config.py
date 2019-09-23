import os
from collections import OrderedDict

import numpy as np
import yaml

from utility.ordered_easydict import OrderedEasyDict as edict


def config_gru_fms(height, strides):
    gru_fms = [height]
    for i, s in enumerate(strides):
        gru_fms.append(gru_fms[i] // s)
    return gru_fms[1:]


def config_deconv_infer(height, strides):
    infer_shape = [height]
    for i, s in enumerate(strides[:-1]):
        infer_shape.append(infer_shape[i] // s)
    return infer_shape


__C = edict()
c = __C  # type: edict()


# iterator
__C.DATA_BASE_PATH = os.path.join("/extend", "sz17_data")
__C.REF_PATH = os.path.join(__C.DATA_BASE_PATH, "radarPNG_expand")
__C.TRAIN_DIR_CLIPS = os.path.join(__C.DATA_BASE_PATH, "15-17_clips")
__C.VALID_DIR_CLIPS = os.path.join(__C.DATA_BASE_PATH, "18_clips")

__C.BASE_PATH = os.path.join("/extend", "gru_tf_data")
__C.SAVE_PATH = os.path.join(__C.BASE_PATH, "0512_ebgan")
__C.SAVE_MODEL = os.path.join(__C.SAVE_PATH, "Save")
__C.SAVE_VALID = os.path.join(__C.SAVE_PATH, "Valid")
__C.SAVE_TEST = os.path.join(__C.SAVE_PATH, "Test")
__C.SAVE_SUMMARY = os.path.join(__C.SAVE_PATH, "Summary")
__C.SAVE_METRIC = os.path.join(__C.SAVE_PATH, "Metric")

if not os.path.exists(__C.SAVE_MODEL):
    os.makedirs(__C.SAVE_MODEL)
if not os.path.exists(__C.SAVE_VALID):
    os.makedirs(__C.SAVE_VALID)

__C.RAINY_TRAIN = ['201501010000', '201801010000']
__C.RAINY_VALID = ['201801010000', '201809180000']
__C.RAINY_TEST = ['201805110000', '201806080000']

# train
__C.MAX_ITER = 100001
__C.SAVE_ITER = 5000
__C.VALID_ITER = 5000

__C.SUMMARY_ITER = 50

# project
__C.DTYPE = "single"
__C.NORMALIZE = False
__C.FULL_H = 700
__C.FULL_W = 900
__C.MOVEMENT_THRESHOLD = 3000
__C.H = 480
__C.W = 480

__C.BATCH_SIZE = 2
__C.IN_CHANEL = 1
__C.RNN_CELL = "conv_gru"     # conv_gru st_lstm PredRNN
__C.PRED_RNN_LAYERS = 4
# encoder
# (kernel, kernel, in chanel, out chanel)

__C.CONV_KERNEL = ((7, 7, __C.IN_CHANEL, 8),
               (5, 5, 64, 192),
               (3, 3, 192, 192))
__C.CONV_STRIDE = (5, 3, 2)
__C.ENCODER_GRU_FILTER = (64, 192, 192)
__C.ENCODER_GRU_INCHANEL = (8, 192, 192)
__C.DOWN_SAMPLE_TYPE = "conv"  # conv incept
# decoder
# (kernel, kernel, out chanel, in chanel)
__C.DECONV_KERNEL = ((7, 7, 8, 64),
                 (5, 5, 64, 192),
                 (4, 4, 192, 192))
__C.DECONV_STRIDE = (5, 3, 2)
__C.DECODER_GRU_FILTER = (64, 192, 192)
__C.DECODER_GRU_INCHANEL = (64, 192, 192)
__C.UP_SAMPLE_TYPE = "deconv"  # deconv incept
# Encoder Forecaster
__C.IN_SEQ = 5
__C.OUT_SEQ = 10

__C.LR = 0.0001

__C.RESIDUAL = False
__C.SEQUENCE_MODE = False

# RNN
__C.I2H_KERNEL = [3, 3, 3]
__C.H2H_KERNEL = [5, 5, 3]

# EVALUATION
__C.ZR_a = 58.53
__C.ZR_b = 1.56

__C.EVALUATION_THRESHOLDS = (12.9777173087837, 28.577717308783704, 33.27378524114181, 40.71687681476854)

__C.USE_BALANCED_LOSS = False
__C.THRESHOLDS = [0.5, 2, 5, 10, 30]
__C.BALANCING_WEIGHTS = [1, 1, 2, 5, 10, 30]

__C.TEMPORAL_WEIGHT_TYPE = "same"
__C.TEMPORAL_WEIGHT_UPPER = 5

# LOSS
__C.L1_LAMBDA = 0.
__C.L2_LAMBDA = 1.0
__C.GDL_LAMBDA = 0.

# PREDICTION
__C.PREDICT_LENGTH = 20
__C.PREDICTION_H = 900
__C.PREDICTION_W = 900

# Discriminator
__C.ADVERSARIAL = True
__C.ADV_LAMBDA = 1.0
__C.MARGIN = 1
__C.ADV_INVOLVE = 0


def _merge_two_config(user_cfg, default_cfg):
    """ Merge user's config into default config dictionary, clobbering the
        options in b whenever they are also specified in a.
        Need to ensure the type of two val under same key are the same
        Do recursive merge when encounter hierarchical dictionary
    """
    if type(user_cfg) is not edict:
        return
    for key, val in user_cfg.items():
        # Since user_cfg is a sub-file of default_cfg
        if not key in default_cfg:
            raise KeyError('{} is not a valid config key'.format(key))

        if (type(default_cfg[key]) is not type(val) and
                default_cfg[key] is not None):
            if isinstance(default_cfg[key], np.ndarray):
                val = np.array(val, dtype=default_cfg[key].dtype)
            else:
                raise ValueError(
                     'Type mismatch ({} vs. {}) '
                     'for config key: {}'.format(type(default_cfg[key]),
                                                 type(val), key))
        # Recursive merge config
        if type(val) is edict:
            try:
                _merge_two_config(user_cfg[key], default_cfg[key])
            except:
                print('Error under config key: {}'.format(key))
                raise
        else:
            default_cfg[key] = val


def cfg_from_file(file_name, target=__C):
    """ Load a config file and merge it into the default options.
    """
    import yaml
    with open(file_name, 'r') as f:
        print('Loading YAML config file from %s' %f)
        yaml_cfg = edict(yaml.load(f))
    _merge_two_config(yaml_cfg, target)


def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items(), flow_style=False)

    def _ndarray_representer(dumper, data):
        return dumper.represent_list(data.tolist())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    OrderedDumper.add_representer(edict, _dict_representer)
    OrderedDumper.add_representer(np.ndarray, _ndarray_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def save_cfg(dir_path, source=__C):
    cfg_count = 0
    file_path = os.path.join(dir_path, 'cfg%d.yml' %cfg_count)
    while os.path.exists(file_path):
        cfg_count += 1
        file_path = os.path.join(dir_path, 'cfg%d.yml' % cfg_count)
    with open(file_path, 'w') as f:
        print("Save YAML config file to %s" %file_path)
        ordered_dump(source, f, yaml.SafeDumper, default_flow_style=None)


def load_config(file_name):
    import yaml
    with open(file_name, 'r') as f:
        print('Loading YAML config file from %s' %f)
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg



if __name__ == '__main__':
    cfg_from_file("/extend/gru_tf_data/0923_inception/cfg0.yml")
    save_cfg("/extend/gru_tf_data/0923_inception")
