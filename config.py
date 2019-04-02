import os


#iterator
DATA_BASE_PATH = os.path.join("/extend", "sz17_data")
REF_PATH = os.path.join(DATA_BASE_PATH, "radarPNG_expand")

BASE_PATH = os.path.join("/extend", "gru_tf_data")
SAVE_PATH = os.path.join(BASE_PATH, "0402_test")
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

#train
MAX_ITER = 100001
SAVE_ITER = 5000
VALID_ITER = 5000

SUMMARY_ITER = 10

# project
DTYPE = "single"
NORMALIZE = False

H = 900
W = 900

BATCH_SIZE = 2
IN_CHANEL = 1

# Encoder Forecaster
IN_SEQ = 5
OUT_SEQ = 10

LR = 0.0001

RESIDUAL = False
SEQUENCE_MODE = False

GRU_FMS = (180, 60, 30)

# encoder
# (kernel, stride, in chanel, out chanel)

CONV_FMS = ((7, 7, IN_CHANEL, 8),
            (5, 5, 64, 192),
            (3, 3, 192, 192))
CONV_STRIDE = (5, 3, 2)
ENCODER_GRU_FILTER = (64, 192, 192)
ENCODER_GRU_INCHANEL = (8, 192, 192)
# decoder
# (kernel, kernel, out chanel, in chanel)
DECONV_FMS = ((7, 7, 8, 64),
              (5, 5, 64, 192),
              (4, 4, 192, 192))
DECONV_STRIDE = (5, 3, 2)
DECODER_GRU_FILTER = (64, 192, 192)
DECODER_GRU_INCHANEL = (64, 192, 192)
DECODER_INFER_SHAPE = (900, 180, 60)

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
