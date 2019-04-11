import os


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


# iterator
DATA_BASE_PATH = os.path.join("/extend", "sz17_data")
REF_PATH = os.path.join(DATA_BASE_PATH, "radarPNG_expand")
TRAIN_DIR_CLIPS = os.path.join(DATA_BASE_PATH, "15-17_clips")
VALID_DIR_CLIPS = os.path.join(DATA_BASE_PATH, "18_clips")

BASE_PATH = os.path.join("/extend", "gru_tf_data")
SAVE_PATH = os.path.join(BASE_PATH, "0409_small1")
SAVE_MODEL = os.path.join(SAVE_PATH, "Save")
SAVE_VALID = os.path.join(SAVE_PATH, "Valid")
SAVE_TEST = os.path.join(SAVE_PATH, "Test")
SAVE_SUMMARY = os.path.join(SAVE_PATH, "Summary")
SAVE_METRIC = os.path.join(SAVE_PATH, "Metric")

if not os.path.exists(SAVE_MODEL):
    os.makedirs(SAVE_MODEL)
if not os.path.exists(SAVE_VALID):
    os.makedirs(SAVE_VALID)

RAINY_TRAIN = ['201501010000', '201801010000']
RAINY_VALID = ['201801010000', '201809180000']
RAINY_TEST = ['201805110000', '201806080000']

# train
MAX_ITER = 100001
SAVE_ITER = 5000
VALID_ITER = 50

SUMMARY_ITER = 50

# project
DTYPE = "single"
NORMALIZE = False
FULL_H = 700
FULL_W = 900
MOVEMENT_THRESHOLD = 3000
H = 480
W = 480

BATCH_SIZE = 2
IN_CHANEL = 1

# encoder
# (kernel, kernel, in chanel, out chanel)

CONV_KERNEL = ((7, 7, IN_CHANEL, 8),
               (5, 5, 64, 192),
               (3, 3, 192, 192))
CONV_STRIDE = (5, 3, 2)
ENCODER_GRU_FILTER = (64, 192, 192)
ENCODER_GRU_INCHANEL = (8, 192, 192)
# decoder
# (kernel, kernel, out chanel, in chanel)
DECONV_KERNEL = ((7, 7, 8, 64),
                 (5, 5, 64, 192),
                 (4, 4, 192, 192))
DECONV_STRIDE = (5, 3, 2)
DECODER_GRU_FILTER = (64, 192, 192)
DECODER_GRU_INCHANEL = (64, 192, 192)

# Encoder Forecaster
IN_SEQ = 5
OUT_SEQ = 10

LR = 0.0001

RESIDUAL = False
SEQUENCE_MODE = False

# RNN
I2H_KERNEL = [3, 3, 3]
H2H_KERNEL = [5, 5, 3]

# EVALUATION
ZR_a = 58.53
ZR_b = 1.56

EVALUATION_THRESHOLDS = (0, 5, 10, 30)

USE_BALANCED_LOSS = False
THRESHOLDS = [0.5, 2, 5, 10, 30]
BALANCING_WEIGHTS = [1, 1, 2, 5, 10, 30]

TEMPORAL_WEIGHT_TYPE = "same"
TEMPORAL_WEIGHT_UPPER = 5

# LOSS
L1_LAMBDA = 0
L2_LAMBDA = 1.0
GDL_LAMBDA = 0

# PREDICTION
PREDICT_LENGTH = 20
PREDICTION_H = 900
PREDICTION_W = 900


if __name__ == '__main__':
    print(config_gru_fms(PREDICTION_H, CONV_STRIDE))
    print(config_deconv_infer(PREDICTION_H, DECONV_STRIDE))