from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.BACKBONE_EMB_SIZE = 2048

_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Path to tranformers pre-train path
_C.MODEL.TRANS_PRETRAIN_PATH = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False
# Use ImageNet pretrained model to initialize backbone or use 'self' trained
# model to initialize the whole model
# Options: True | False
_C.MODEL.PRETRAINED = True
    # Create centroids
_C.MODEL.USE_CENTROIDS = False
# Ensures images to build centroids during retrieval
# do not come from the same camera as the query
_C.MODEL.KEEP_CAMID_CENTROIDS = True
# Set True if Pre-traing path points to previously trained/aborted model
_C.MODEL.RESUME_TRAINING = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = "market1501"
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = "/home/data"
# Path to json train file for datasets that require it
_C.DATASETS.JSON_TRAIN_PATH = ""

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = "random_identity"
_C.DATALOADER.TRANS_SAMPLER = "softmax"
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 4
# Whether to drop last not full batch
_C.DATALOADER.DROP_LAST = True
# Whether to use resampling in case when number of samples < DATALOADER.NUM_INSTANCE:
# True for Baseline. False for CTLModel
_C.DATALOADER.USE_RESAMPLING = True

# ---------------------------------------------------------------------------- #
# SOLVER
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 120
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.5
# Function used to compute distance (euclidean or cosine for now)
_C.SOLVER.DISTANCE_FUNC = "euclidean"
# # Margin of cluster
_C.SOLVER.CLUSTER_MARGIN = 0.3
# # Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# # Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
# name of LR scheduler
_C.SOLVER.LR_SCHEDULER_NAME = "multistep_lr"
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (40, 70)
_C.SOLVER.LR_STEPS = (40, 70)
# warm up factor
_C.SOLVER.USE_WARMUP_LR = True
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 5
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
# Metric name for checkpointing best model
_C.SOLVER.MONITOR_METRIC_NAME = "mAP"
# Metric value mode used for checkpointing (max, min, auto)
_C.SOLVER.MONITOR_METRIC_MODE = "max"
# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 50
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
# 'dp', 'ddp', 'ddp2', 'ddp_spawn' - see pytorch lighning options
_C.SOLVER.DIST_BACKEND = "ddp"
# Losses weights
# Weight of classification loss on query vectors
_C.SOLVER.QUERY_XENT_WEIGHT = 1.0
# Weight of contrastive loss on query vectors
_C.SOLVER.QUERY_CONTRASTIVE_WEIGHT = 1.0
# Weight of contrastive loss on centroids-query vectors
_C.SOLVER.CENTROID_CONTRASTIVE_WEIGHT = 1.0
# Whether to use automatic Python Lightning optimization
_C.SOLVER.USE_AUTOMATIC_OPTIM = False
_C.SOLVER.PCA_K=256
# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
# # Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = True
# Only run test
_C.TEST.ONLY_TEST = False
# If to visualize rank results
_C.TEST.VISUALIZE = "no"
# What top-k results to rank
_C.TEST.VISUALIZE_TOPK = 10
# Max number of query images plotted
_C.TEST.VISUALIZE_MAX_NUMBER = 1000000

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
# ---------------------------------------------------------------------------- #
# MISC
# ---------------------------------------------------------------------------- #
# Ids of GPU devices to use during training, especially when using ddp backend
_C.GPU_IDS = [0]
# Log root directory
_C.LOG_DIR = "logs"
# Whether to use mixed precision
_C.USE_MIXED_PRECISION = True
# If output dir is specified it overrides automatic output path creation
_C.OUTPUT_DIR = ""

# ---------------------------------------------------------------------------- #
# REPORDUCIBLE EXPERIMENTS
# ---------------------------------------------------------------------------- #
# Whether to seed everything
_C.REPRODUCIBLE = False
# Number of runs with seeded generators
_C.REPRODUCIBLE_NUM_RUNS = 3
# Seed to start with
_C.REPRODUCIBLE_SEED = 0
