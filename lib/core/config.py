from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

config = edict()

config.WORKERS = 2
config.LOG_DIR = ''
config.MODEL_DIR = ''
config.RESULT_DIR = ''
config.DATA_DIR = ''
config.JSON_DIR = ''
config.VERBOSE = False
config.TAG = ''
config.WORD_MASK_RATIO = 0


# CUDNN related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# CAEM related params
config.CAEM = edict()
config.CAEM.FRAME_MODULE = edict()
config.CAEM.FRAME_MODULE.NAME = ''
config.CAEM.FRAME_MODULE.PARAMS = None
config.CAEM.PROP_MODULE = edict()
config.CAEM.PROP_MODULE.NAME = ''
config.CAEM.PROP_MODULE.PARAMS = None
config.CAEM.FUSION_MODULE = edict()
config.CAEM.FUSION_MODULE.NAME = ''
config.CAEM.FUSION_MODULE.PARAMS = None
config.CAEM.MAP_MODULE = edict()
config.CAEM.MAP_MODULE.NAME = ''
config.CAEM.MAP_MODULE.PARAMS = None
config.CAEM.VLBERT_MODULE = edict()
config.CAEM.VLBERT_MODULE.NAME = ''
config.CAEM.VLBERT_MODULE.PARAMS = None
config.CAEM.PRED_INPUT_SIZE = 512

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = ''
config.MODEL.CHECKPOINT = '' # The checkpoint for the best performance

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.NAME = ''
config.DATASET.MODALITY = ''
config.DATASET.VIS_INPUT_TYPE = ''
config.DATASET.NO_VAL = False
config.DATASET.NO_IID = False
config.DATASET.NO_OOD = False
config.DATASET.BIAS = 0
config.DATASET.NUM_SAMPLE_CLIPS = 256
config.DATASET.TARGET_STRIDE = 16
config.DATASET.DOWNSAMPLING_STRIDE = 16
config.DATASET.SPLIT = ''
config.DATASET.NORMALIZE = False
config.DATASET.RANDOM_SAMPLING = False
config.DATASET.RANDOM_FLAG = 0


# train
config.TRAIN = edict()
config.TRAIN.LR = 0.001
config.TRAIN.WEIGHT_DECAY = 0
config.TRAIN.FACTOR = 0.8
config.TRAIN.PATIENCE = 20
config.TRAIN.MAX_EPOCH = 20
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.SHUFFLE = True
config.TRAIN.CONTINUE = False
config.TRAIN.SAVE_CHECKPOINT = True
config.TRAIN.SAVE_ALL_CHECKPOINT = True
config.TRAIN.EVAL_METRIC = 'dR'
config.TRAIN.WEIGHT = 0
config.TRAIN.CTF = True

config.LOSS = edict()
config.LOSS.NAME = 'bce_loss'
config.LOSS.PARAMS = None
config.LOSS.USING_NEG = False

# test
config.TEST = edict()
config.TEST.RECALL = []
config.TEST.TIOU = []
config.TEST.NMS_THRESH = 0.4
config.TEST.INTERVAL = 0.25
config.TEST.EVAL_TRAIN = False
config.TEST.BATCH_SIZE = 1
config.TEST.TOP_K = 10
config.TEST.BEST_METRIC = 'mIoU'
config.TEST.EVAL_METRIC = 'dR'
config.TEST.WEIGHT = 0


def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k],v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))

def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
