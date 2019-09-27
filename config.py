import os
from collections import OrderedDict

import numpy as np
import yaml

from utility.ordered_easydict import OrderedEasyDict as edict


def config_gru_fms(height, strides):
    """
    用来计算每层gru feature map 的大小
    :param height: 输入ref的长宽（要求是正方形）
    :param strides: 每层卷积的步长
    :return: 
    """
    gru_fms = [height]
    for i, s in enumerate(strides):
        gru_fms.append(gru_fms[i] // s)
    return gru_fms[1:]


def config_deconv_infer(height, strides):
    """
    推导转置卷积的输出形状
    :param height: 输入ref的长宽（要求是正方形）
    :param strides: 步长
    :return: 
    """
    infer_shape = [height]
    for i, s in enumerate(strides[:-1]):
        infer_shape.append(infer_shape[i] // s)
    return infer_shape


__C = edict()
c = __C  # type: edict()


# iterator
__C.DATA_BASE_PATH = os.path.join("/extend", "sz17_data")               # 数据集整体基路径
__C.REF_PATH = os.path.join(__C.DATA_BASE_PATH, "radarPNG_expand")      # 输入ref数据文件夹
__C.TRAIN_DIR_CLIPS = os.path.join(__C.DATA_BASE_PATH, "15-17_clips")   # 如果使用经过剪裁的小图片做训练，输入文件夹路径
__C.VALID_DIR_CLIPS = os.path.join(__C.DATA_BASE_PATH, "18_clips")      # 小图片作训练时，Valid 数据集路径

__C.BASE_PATH = os.path.join("/extend", "gru_tf_data")                  # 输出结果基路径
__C.SAVE_PATH = os.path.join(__C.BASE_PATH, "0512_ebgan")               # 本次试验结果存放路径
__C.SAVE_MODEL = os.path.join(__C.SAVE_PATH, "Save")                    # 模型保存路径
__C.SAVE_VALID = os.path.join(__C.SAVE_PATH, "Valid")                   # Valid 结果保存路径
__C.SAVE_TEST = os.path.join(__C.SAVE_PATH, "Test")                     # 测试结果保存路径
__C.SAVE_SUMMARY = os.path.join(__C.SAVE_PATH, "Summary")               # Tensorflow Summary 结果保存路径
__C.SAVE_METRIC = os.path.join(__C.SAVE_PATH, "Metric")                 # Evaluation 结果保存路径

if not os.path.exists(__C.SAVE_MODEL):
    os.makedirs(__C.SAVE_MODEL)
if not os.path.exists(__C.SAVE_VALID):
    os.makedirs(__C.SAVE_VALID)

__C.RAINY_TRAIN = ['201501010000', '201801010000']                      # 训练时间段
__C.RAINY_VALID = ['201801010000', '201809180000']                      # Valid 时间段
__C.RAINY_TEST = ['201805110000', '201806080000']                       # Test 时间段

# train
__C.MAX_ITER = 100001                                                   # 最大训练轮数
__C.SAVE_ITER = 5000                                                    # 保存间隔
__C.VALID_ITER = 5000                                                   # Valid 间隔
__C.SUMMARY_ITER = 50                                                   # 输出 Summary 间隔

# project
__C.DTYPE = "single"                                                    # 模型数值精度（单精度浮点数 single 还是半精度 half）
__C.NORMALIZE = False                                                   # 是否对输入数据进行归一化
__C.FULL_H = 700                                                        # ref 实际的高
__C.FULL_W = 900                                                        # ref 实际的宽
__C.MOVEMENT_THRESHOLD = 3000                                           # 作小图片剪切时，每张图像上有数据点的最低阈值
__C.H = 480                                                             # iterator 产生图像的高
__C.W = 480                                                             # iterator 产生图像的宽

__C.BATCH_SIZE = 2
__C.IN_CHANEL = 1                                                       # 输入ref的通道数（高度层数）
__C.RNN_CELL = "conv_gru"                                               # RNN 模型类型：conv_gru st_lstm PredRNN
__C.PRED_RNN_LAYERS = 4                                                 # 仅 PredRNN 时有效，st lstm 堆叠的层数

# encoder
# (kernel, kernel, in chanel, out chanel)
__C.CONV_KERNEL = ((7, 7, __C.IN_CHANEL, 8),                            # 每层卷积卷积核的大小
               (5, 5, 64, 192),                                         # 如果是 incept 则需要为二维 list
               (3, 3, 192, 192))
__C.CONV_STRIDE = (5, 3, 2)                                             # 每层卷积的步长
__C.ENCODER_GRU_FILTER = (64, 192, 192)                                 # 每层 RNN 的 filter 数
__C.ENCODER_GRU_INCHANEL = (8, 192, 192)                                # 每层 RNN 的输入通道数
__C.DOWN_SAMPLE_TYPE = "conv"                                           # encoder 中 下采样种类 incept conv

# decoder
# (kernel, kernel, out chanel, in chanel)
__C.DECONV_KERNEL = ((7, 7, 8, 64),                                     # 每层转置卷积核的大小
                 (5, 5, 64, 192),                                       # 如果是 incept 则需要为二维 list
                 (4, 4, 192, 192))
__C.DECONV_STRIDE = (5, 3, 2)                                           # 每层转置卷积的步长
__C.DECODER_GRU_FILTER = (64, 192, 192)                                 # 每层 RNN 的 filter 数
__C.DECODER_GRU_INCHANEL = (64, 192, 192)                               # 每层 RNN 的输入通道数
__C.UP_SAMPLE_TYPE = "deconv"                                           # decoder 中 下采样种类 incept deconv

# Encoder Forecaster
__C.IN_SEQ = 5                                                          # 模型输入序列长度
__C.OUT_SEQ = 10                                                        # 模型输出序列长度（训练时）

__C.LR = 0.0001                                                         # Learning Rate

__C.RESIDUAL = False                                                    # 没用
__C.SEQUENCE_MODE = True                                                # 没用

# RNN
__C.I2H_KERNEL = [3, 3, 3]                                              # 每层 RNN 内部各个卷积卷积核大小
__C.H2H_KERNEL = [5, 5, 3]                                              # conv gru 时使用，每层 h2h 核大小

# EVALUATION
__C.ZR_a = 58.53                                                        # ZR 关系中的 a
__C.ZR_b = 1.56                                                         # ZR 关系中的 b

# 用作 evaluation 时的 threshold，[0.5mm, 5mm, 10mm, 30mm] 经过zr转换为 dBZ之后
__C.EVALUATION_THRESHOLDS = (12.9777173087837, 28.577717308783704, 33.27378524114181, 40.71687681476854)

__C.USE_BALANCED_LOSS = False                                           # 是否根据预测值的大小设置权重
__C.THRESHOLDS = [0.5, 2, 5, 10, 30]                                    # 分段阈值
__C.BALANCING_WEIGHTS = [1, 1, 2, 5, 10, 30]                            # 权重大小

__C.TEMPORAL_WEIGHT_TYPE = "same"                                       # 输出时间序列中每个时段权重
__C.TEMPORAL_WEIGHT_UPPER = 5                                           # 上界

# LOSS
__C.L1_LAMBDA = 0.                                                      # 损失函数中 L1 系数
__C.L2_LAMBDA = 1.0                                                     # 损失函数中 L2 系数
__C.GDL_LAMBDA = 0.                                                     # 损失函数中 GDL 系数

# PREDICTION
__C.PREDICT_LENGTH = 20                                                 # 预测时输出序列长度
__C.PREDICTION_H = 900                                                  # 预测和训练时，模型对于图像大小的要求，高
__C.PREDICTION_W = 900                                                  # 预测和训练时，模型对于图像大小的要求，宽

# Discriminator EBGAN
__C.ADVERSARIAL = True                                                  # 是否加入 EBGAN
__C.ADV_LAMBDA = 1.0                                                    # EBGAN loss 的系数
__C.MARGIN = 1                                                          # EBGAN 参数 factor
__C.ADV_INVOLVE = 0                                                     # 在多少轮时 EBGAN 进行介入
__C.DIS_FMS = [7, 5, 3]

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
